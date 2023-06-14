from torch_geometric.loader import DataLoader


class MultiBatchLoader(DataLoader):
    def __init__(self, dataset, batch_sampler):
        super(MultiBatchLoader, self).__init__(dataset, batch_sampler=batch_sampler)

    def __iter__(self):
        for multi_indices in self.batch_sampler:
            multi_batch = []
            for batch_indices in multi_indices:
                batch = [self.dataset[i] for i in batch_indices]
                multi_batch.append(self.collate_fn(batch))
            yield multi_batch
