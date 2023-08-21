import torch as t
from federatedml.nn.homo.trainer.fedavg_trainer import FedAVGTrainer
from federatedml.framework.homo.aggregator.secure_aggregator import SecureAggregatorClient as SecureAggClient
from federatedml.framework.homo.aggregator.secure_aggregator import SecureAggregatorServer as SecureAggServer
from federatedml.util import LOGGER
from fate_llm.model_zoo.offsite_tuning.offsite_tuning_model import OffsiteTuningSubModel, OffsiteTuningMainModel
from federatedml.util import consts
from federatedml.nn.backend.utils import deepspeed_util
from federatedml.nn.backend.utils import distributed_util


def count_parameters(model: t.nn.Module):
    return sum(p.numel() for p in model.parameters())


def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class OffsiteTuningTrainer(FedAVGTrainer):

    def __init__(self, epochs=10, batch_size=512,  # training parameter
                 early_stop=None, tol=0.0001,  # early stop parameters
                 secure_aggregate=True, weighted_aggregation=True, aggregate_every_n_epoch=None,  # federation
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
            epochs=epochs, batch_size=batch_size,
            early_stop=early_stop, tol=tol,
            secure_aggregate=secure_aggregate, weighted_aggregation=weighted_aggregation, aggregate_every_n_epoch=aggregate_every_n_epoch,
            cuda=cuda,
            pin_memory=pin_memory, shuffle=shuffle, data_loader_worker=data_loader_worker,
            validation_freqs=validation_freqs,
            checkpoint_save_freqs=checkpoint_save_freqs,
            task_type=task_type,
            save_to_local_dir=save_to_local_dir,
            collate_fn=collate_fn,
            collate_fn_params=collate_fn_params
        )

        self.need_aggregate = need_aggregate

    def on_loop_begin_client(self):

        if not isinstance(self.model, OffsiteTuningSubModel):
            raise ValueError('Client must provide a model subclassing "OffsiteTuningSubModel" in the offsite-tuning trainer, but got {}'.format(type(self.model)))

        if distributed_util.is_distributed(): 
            if not distributed_util.is_rank_0():
                self._share_model()
                return
            else: # unwarp model
                model: OffsiteTuningSubModel = self.model  # unwrap model here
        else:
            model: OffsiteTuningSubModel = self.model

        if self.fed_mode:
            # receive parameters from model provider and load emulator, adapter
            client_agg: SecureAggregatorClient = self.client_agg
            param = client_agg.get('sub_model_parameter')
            model.load_submodel_weights(param[0])
            LOGGER.info('loaded weights keys are {}'.format(param[0].keys()))

            if distributed_util.is_distributed() and distributed_util.is_rank_0():  # scatter model weights
                self._share_model()
        else:
            pass
        
            LOGGER.info('adapter parameters num: {}'.format(count_parameters(model.get_adapter_top()) + count_parameters(model.get_adapter_bottom())))
            LOGGER.info('trainable parameters num {}'.format(count_trainable_parameters(model)))

    def on_loop_begin_server(self):

        if self.model is None:
            raise ValueError('Server must provide a main model in the offsite-tuning trainer, got None model, \
                               please set server_init to True and provide the model config')

        if not isinstance(self.model, OffsiteTuningMainModel):
            raise ValueError('Server must provide a model subclassing "OffsiteTuningMainModel" in the offsite-tuning trainer, but got {}'.format(type(self.model)))

        model: OffiteTuningMainModel = self.model
        sub_model_state_dict = model.get_submodel_weights()
        server_agg: SecureAggregatorServer = self.server_agg
        server_agg.broadcast(sub_model_state_dict, suffix='sub_model_parameter')

        LOGGER.info('adapter parameters num: {}'.format(count_parameters(model.get_adapter_top()) + count_parameters(model.get_adapter_bottom())))
        LOGGER.info('emulator parameters num: {}'.format(count_parameters(model.get_emulator())))

    def on_loop_end_client(self):

        if self.fed_mode:
            if distributed_util.is_distributed():
                if distributed_util.is_rank_0():
                    pass  # unwrap model here
                else:
                    model: OffsiteTuningSubModel = self.model  # unwarp model here
            else:
                model: OffsiteTuningSubModel = self.model
            sub_model_state_dict = model.get_submodel_weights()
            client_agg = self.client_agg
            client_agg.send(sub_model_state_dict, suffix='final_sub_model_parameter') 
        else:
            return

    def on_loop_end_server(self):
        
        model: OffsiteTuningMainModel = self.model
        server_agg = self.server_agg
        sub_model_state_dict = server_agg.collect(suffix='final_sub_model_parameter')[0]
        model.load_submodel_weights(sub_model_state_dict)
        LOGGER.info('load trained adapters parameters from clients')

    def _client_sends_data(self, epoch_idx, epoch_loss, cur_agg_round):
        if self.need_aggregate:
            return super()._client_sends_data(epoch_idx, epoch_loss, cur_agg_round)
        else:
            return False

    def _server_aggregates_data(self, epoch_idx, check_converge, converge_func):
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
                if len(self.party_id_list) == 1: # guest only:
                    clients = (consts.GUEST, )
                else:
                    clients = (consts.GUEST, consts.HOST)
                client_agg = SecureAggClient(
                    self.secure_aggregate, aggregate_weight=sample_num, communicate_match_suffix=self.comm_suffix, clients=clients)
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
        if len(self.party_id_list) == 1: # guest only:
                    clients = (consts.GUEST, )
        else:
            clients = (consts.GUEST, consts.HOST)
        self.server_agg = SecureAggServer(self.secure_aggregate, communicate_match_suffix=self.comm_suffix, clients=clients)

        self.on_loop_begin_server()
        # aggregate and broadcast models
        for i in range(self.epochs):

            need_stop = self._server_aggregates_data(i, check_converge, converge_func)
            if need_stop:
                break
                
        self.on_loop_end_server()
        LOGGER.info('server aggregation process done')
        if self.model is not None:
            if self.save_to_local_dir:
                self.local_save(model=self.model, epoch_idx=i, converge_status=need_stop)
            else:
                self.save(model=self.model, epoch_idx=i, converge_status=need_stop)
            LOGGER.info('sever side model saved')