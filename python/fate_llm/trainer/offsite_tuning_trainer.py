#
#  Copyright 2019 The FATE Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
import torch as t
from federatedml.nn.homo.trainer.fedavg_trainer import FedAVGTrainer
from federatedml.framework.homo.aggregator.secure_aggregator import SecureAggregatorClient as SecureAggClient
from federatedml.framework.homo.aggregator.secure_aggregator import SecureAggregatorServer as SecureAggServer
from federatedml.util import LOGGER
from fate_llm.model_zoo.offsite_tuning.offsite_tuning_model import OffsiteTuningSubModel, OffsiteTuningMainModel
from federatedml.util import consts
from federatedml.nn.backend.utils import deepspeed_util
from federatedml.nn.backend.utils import distributed_util
import torch.distributed as dist
from federatedml.optim.convergence import converge_func_factory



def count_parameters(model: t.nn.Module):
    return sum(p.numel() for p in model.parameters())


def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class OffsiteTuningTrainer(FedAVGTrainer):

    def __init__(self, epochs=10, batch_size=512,  # training parameter
                 early_stop=None, tol=0.0001,  # early stop parameters
                 secure_aggregate=False, weighted_aggregation=True, aggregate_every_n_epoch=None,  # federation, offsite tuning need to aggregate large model, default is False
                 cuda=None,
                 pin_memory=True, shuffle=True, data_loader_worker=0,  # GPU & dataloader
                 validation_freqs=None,  # validation configuration
                 checkpoint_save_freqs=None,  # checkpoint configuration
                 task_type='auto',  # task type
                 save_to_local_dir=False,  # save model to local path
                 collate_fn=None,
                 collate_fn_params=None,
                 need_aggregate=False
                 ):

        super().__init__(
            epochs=epochs,
            batch_size=batch_size,
            early_stop=early_stop,
            tol=tol,
            secure_aggregate=secure_aggregate,
            weighted_aggregation=weighted_aggregation,
            aggregate_every_n_epoch=aggregate_every_n_epoch,
            cuda=cuda,
            pin_memory=pin_memory,
            shuffle=shuffle,
            data_loader_worker=data_loader_worker,
            validation_freqs=validation_freqs,
            checkpoint_save_freqs=checkpoint_save_freqs,
            task_type=task_type,
            save_to_local_dir=save_to_local_dir,
            collate_fn=collate_fn,
            collate_fn_params=collate_fn_params)

        self.need_aggregate = need_aggregate
        self.model_transvar = None


    def _send_submodel_weights(self, state_dict, send_func, suffix='start'):
        from fate_arch.session import computing_session as session
        emulator = state_dict['emulator']
        adapter_top = state_dict['adapter_top']
        adapter_bottom = state_dict['adapter_bottom']
        tb1 = session.parallelize([(key, value) for key, value in emulator.items()], include_key=True, partition=4)
        tb2 = session.parallelize([(key, value) for key, value in adapter_top.items()], include_key=True, partition=4)
        tb3 = session.parallelize([(key, value) for key, value in adapter_bottom.items()], include_key=True, partition=4)
        state_dict.pop('emulator', None)
        state_dict.pop('adapter_top', None)
        state_dict.pop('adapter_bottom', None)
        tb4 = session.parallelize([(key, value) for key, value in state_dict.items()], include_key=True, partition=4)
        send_func(
            tb1,
            suffix='emulator_'+suffix)
        send_func(
            tb2,
            suffix='adapter_top_'+suffix)
        send_func(
            tb3,
            suffix='adapter_bottom_'+suffix)
        send_func(
            tb4,
            suffix='other_param_'+suffix)


    def _get_submodel_weights(self, get_func, suffix='start'):
        
        tb1 = get_func(suffix='emulator_'+suffix)[0]
        tb2 = get_func(suffix='adapter_top_'+suffix)[0]
        tb3 = get_func(suffix='adapter_bottom_'+suffix)[0]
        tb4 = get_func(suffix='other_param_'+suffix)[0]

        got_state_dict = {}
        got_state_dict['emulator'] = dict(tb1.collect())
        got_state_dict['adapter_top'] = dict(tb2.collect())
        got_state_dict['adapter_bottom'] = dict(tb3.collect())
        other_param = dict(tb4.collect())
        got_state_dict.update(other_param)

        return got_state_dict


    def on_loop_begin_client(self):
        
        unwarp_model = self.unwrap_model(self.model)
        if not isinstance(unwarp_model, OffsiteTuningSubModel):
            raise ValueError(
                'Client must provide a model subclassing "OffsiteTuningSubModel" in the offsite-tuning trainer, but got {}'.format(
                    type(
                        unwarp_model)))

        model: OffsiteTuningSubModel = unwarp_model

        if self.fed_mode:

            if (distributed_util.is_distributed() and distributed_util.is_rank_0()) or (not distributed_util.is_distributed()):
                # receive parameters from model provider and load emulator, adapter
                ret = self._get_submodel_weights(self.model_transvar.server_to_client.get, suffix='start')
                LOGGER.info('loaded weights keys are {}'.format(ret.keys()))
                # client_agg: SecureAggregatorClient = self.client_agg
                # param = client_agg.get('sub_model_parameter')
                model.load_submodel_weights(ret)

            if distributed_util.is_distributed(): 
                self._share_model(sync_trainable_only=False)
                # reinitalize deepspeed
                deepspeed_util.init_deepspeed_env(self._ds_config)
                model = self.unwrap_model(self.model)
                self._model, self._optimizer = deepspeed_util.deepspeed_init(model, self._ds_config)
                if deepspeed_util.is_zero3(self._ds_config):
                    self._model.train()

        LOGGER.info(
            'adapter parameters num: {}'.format(
                count_parameters(
                    model.get_adapter_top()) +
                count_parameters(
                    model.get_adapter_bottom())))
        LOGGER.info(
            'trainable parameters num {}'.format(
                count_trainable_parameters(model)))

    def on_loop_begin_server(self):
        
        if self.model is None:
            raise ValueError(
                'Server must provide a main model in the offsite-tuning trainer, got None model, \
                               please set server_init to True and provide the model config')

        unwrap_model = self.unwrap_model(self.model)
        if not isinstance(unwrap_model, OffsiteTuningMainModel):
            raise ValueError(
                'Server must provide a model subclassing "OffsiteTuningMainModel" in the offsite-tuning trainer, but got {}'.format(
                    type(
                        unwrap_model)))

        model: OffsiteTuningMainModel = unwrap_model
        sub_model_state_dict = model.get_submodel_weights()
        self._send_submodel_weights(sub_model_state_dict, self.model_transvar.server_to_client.remote, suffix='start')
        # server_agg: SecureAggregatorServer = self.server_agg
        # server_agg.broadcast(
        #     sub_model_state_dict,
        #     suffix='sub_model_parameter')

        LOGGER.info(
            'adapter parameters num: {}'.format(
                count_parameters(
                    model.get_adapter_top()) +
                count_parameters(
                    model.get_adapter_bottom())))
        LOGGER.info(
            'emulator parameters num: {}'.format(
                count_parameters(
                    model.get_emulator())))

    def on_loop_end_client(self):

        if self.fed_mode:
            if (distributed_util.is_distributed() and distributed_util.is_rank_0()) or (not distributed_util.is_distributed()):
                model: OffsiteTuningSubModel = self.unwrap_model(self.model) 
                sub_model_state_dict = model.get_submodel_weights()
                # client_agg = self.client_agg
                # client_agg.send(
                #     sub_model_state_dict,
                #     suffix='final_sub_model_parameter')
                self._send_submodel_weights(sub_model_state_dict, self.model_transvar.client_to_server.remote, suffix='end')

    def on_loop_end_server(self):
    
        model: OffsiteTuningMainModel = self.model
        ret_state_dict = self._get_submodel_weights(self.model_transvar.client_to_server.get, suffix='end')
        model.load_submodel_weights(ret_state_dict)
        # server_agg = self.server_agg
        # sub_model_state_dict = server_agg.collect(
        #     suffix='final_sub_model_parameter')[0]
        # model.load_submodel_weights(sub_model_state_dict)


    def _client_sends_data(self, epoch_idx, epoch_loss, cur_agg_round):
        if self.need_aggregate:
            return super()._client_sends_data(epoch_idx, epoch_loss, cur_agg_round)
        else:
            return False

    def _server_aggregates_data(
            self,
            epoch_idx,
            check_converge,
            converge_func):
        if self.need_aggregate:
            return super()._server_aggregates_data(epoch_idx, check_converge, converge_func)
        else:
            return False

    def _init_aggregator(self, train_set):
        # compute round to aggregate
        cur_agg_round = 0
        if self.aggregate_every_n_epoch is not None:
            aggregate_round = self.epochs // self.aggregate_every_n_epoch
        else:
            aggregate_round = self.epochs

        # initialize fed avg client
        if self.fed_mode:
            if self.weighted_aggregation:
                sample_num = len(train_set)
            else:
                sample_num = 1.0

            if not distributed_util.is_distributed() or distributed_util.is_rank_0():
                if len(self.party_id_list) == 1:  # guest only:
                    clients = (consts.GUEST, )
                else:
                    clients = (consts.GUEST, consts.HOST)
                client_agg = SecureAggClient(
                    self.secure_aggregate,
                    aggregate_weight=sample_num,
                    communicate_match_suffix=self.comm_suffix,
                    clients=clients,
                    lm_aggregate=True
                    )
                # init model transvar
                from federatedml.framework.homo.blocks import CommunicatorTransVar
                self.model_transvar = CommunicatorTransVar(clients=clients, prefix='model', disable_gc=True)
            else:
                client_agg = None
        else:
            client_agg = None

        return client_agg, aggregate_round

    def server_aggregate_procedure(self, extra_data={}):

        # converge status
        check_converge = False
        converge_func = None
        if self.early_stop:
            check_converge = True
            converge_func = converge_func_factory(
                self.early_stop, self.tol).is_converge
            LOGGER.info(
                'check early stop, converge func is {}'.format(converge_func))

        LOGGER.info('server running aggregate procedure')
        if len(self.party_id_list) == 1:  # guest only:
            clients = (consts.GUEST, )
        else:
            clients = (consts.GUEST, consts.HOST)

        self.server_agg = SecureAggServer(
            self.secure_aggregate,
            communicate_match_suffix=self.comm_suffix,
            clients=clients,
            lm_aggregate=True
            )
        from federatedml.framework.homo.blocks import CommunicatorTransVar
        self.model_transvar = CommunicatorTransVar(clients=clients, prefix='model', disable_gc=True)

        self.on_loop_begin_server()
        # aggregate and broadcast models
        for i in range(self.epochs):

            need_stop = self._server_aggregates_data(
                i, check_converge, converge_func)
            if need_stop:
                break

        self.on_loop_end_server()
        LOGGER.info('server aggregation process done')
        if self._model is not None:
            if self.save_to_local_dir:
                self.local_save(
                    model=self.model,
                    epoch_idx=i,
                    converge_status=need_stop)
            else:
                self.save(
                    model=self.model,
                    epoch_idx=i,
                    converge_status=need_stop)
            LOGGER.info('sever side model saved')
