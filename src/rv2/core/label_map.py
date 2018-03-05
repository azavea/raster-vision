class LabelItem(object):
    def __init__(self, id, name):
        self.id = id
        self.name = name


class LabelMap(object):
    def __init__(self, label_items):
        self.label_item_map = {}
        for label_item in label_items:
            self.label_item_map[label_item.id] = label_item

    def get_by_id(self, id):
        return self.label_item_map[id]

    def get_items(self):
        return self.label_item_map.values()

    def __len__(self):
        return len(self.get_items())

    def get_category_index(self):
        category_index = {}
        for label_item in self.get_items():
            category_index[label_item.id] = {
                'id': label_item.id,
                'name': label_item.name
            }
        return category_index
