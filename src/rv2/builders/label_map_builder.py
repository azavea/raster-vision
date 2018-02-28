from rv2.core.label_map import LabelItem, LabelMap


def build(label_items_config):
    label_items = []
    for label_item_config in label_items_config:
        item = LabelItem(label_item_config.id, label_item_config.name)
        label_items.append(item)
    return LabelMap(label_items)
