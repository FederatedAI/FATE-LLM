from federatedml.nn.homo.trainer.fedavg_trainer import FedAVGTrainer


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