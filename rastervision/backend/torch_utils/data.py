class DataBunch():
    def __init__(self, train_ds, train_dl, valid_ds, valid_dl, label_names):
        self.train_ds = train_ds
        self.train_dl = train_dl
        self.valid_ds = valid_ds
        self.valid_dl = valid_dl
        self.label_names = label_names

    def __repr__(self):
        rep = ''
        rep += 'train_ds: {} items\n'.format(len(self.train_ds))
        rep += 'valid_ds: {} items\n'.format(len(self.valid_ds))
        rep += 'label_names: ' + ','.join(self.label_names)
        return rep
