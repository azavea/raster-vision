from rastervision.protos.class_item_pb2 \
    import ClassItem as ClassItemMsg


class ClassItem(object):
    """A class id and associated data."""

    def __init__(self, id: int, name: str = None, color=None):
        """Construct a new ClassItem.

        Color is picked randomly if it is a null value.

        Args:
            id: (int) class id
            name: (string) name of the class
            color: (string) Pillow color code
        """
        self.id = id
        self.name = name
        self.color = color

    def __eq__(self, other):
        if isinstance(other, ClassItem):
            return (self.id == other.id and self.name == other.name
                    and self.color == other.color)
        return False

    def __repr__(self):
        s = 'CLASS ITEM: [{}] {}'.format(self.id, self.name)
        if self.color:
            s += ' ({})'.format(self.color)
        return s

    def to_proto(self):
        return ClassItemMsg(id=self.id, name=self.name, color=self.color)

    @staticmethod
    def from_proto(msg):
        return ClassItem(id=msg.id, name=msg.name, color=msg.color)


class ClassMap(object):
    """A map from class_id to ClassItem."""

    def __init__(self, class_items):
        """Construct a new ClassMap.

        Args:
            class_items: list of ClassItems
        """
        self.class_item_map = {}
        for class_item in class_items:
            self.class_item_map[class_item.id] = class_item

    def get_by_id(self, id):
        """Return a ClassItem by its id.

        Args:
            id: (int) id of class
        """
        return self.class_item_map[id]

    def get_by_name(self, name):
        for item in self.get_items():
            if name == item.name:
                return item
        raise ValueError('{} is not a name in this ClassMap.'.format(name))

    def get_keys(self):
        """Return the keys."""
        return list(self.class_item_map.keys())

    def get_items(self):
        """Return list of ClassItems."""
        return list(self.class_item_map.values())

    def get_class_names(self):
        """Return list of class names sorted by id."""
        sorted_items = sorted(self.get_items(), key=lambda item: item.id)
        return [item.name for item in sorted_items]

    def __len__(self):
        return len(self.get_items())

    def has_all_colors(self):
        for item in self.get_items():
            if not item.color:
                return False
        return True

    def get_category_index(self):
        """Get the corresponding category_index used by TF Object Detection."""
        category_index = {}
        for class_item in self.get_items():
            category_index[class_item.id] = {
                'id': class_item.id,
                'name': class_item.name
            }
        return category_index

    def to_proto(self):
        """Transform a ClassMap into
        a list of ClassItem protobuf messages
        """
        return [item.to_proto() for item in self.get_items()]

    @staticmethod
    def construct_from(classes):
        """Construct ClassMap from a number of different
           representations.

            Args:
                classes: One of the following:
                         - a ClassMap
                         - a list of class names
                         - a list of ClassItem protobuf messages
                         - a list of ClassItems
                         - a dict which maps class names to class ids
                         - a dict which maps class names to a tuple of
                           (class_id, color), where color is a PIL color string.
        """
        result = None
        if type(classes) is ClassMap:
            result = classes
        elif type(classes) is dict:
            item_list = []
            if not len(classes.items()) == 0:
                if type(list(classes.items())[0][1]) is tuple:
                    # This dict already has colors mapped to class ids
                    for name, (class_id, color) in classes.items():
                        item_list.append(ClassItem(class_id, name, color))
                else:
                    # Map items to empty colors
                    for name, class_id in classes.items():
                        item_list.append(ClassItem(class_id, name))
            result = ClassMap(item_list)
        elif type(classes) is list:
            item_list = []
            if not len(classes) == 0:
                if type(classes[0]) is ClassItemMsg:
                    for item in classes:
                        item_list.append(
                            ClassItem(item.id, item.name, item.color))
                elif type(classes[0]) is str:
                    for i, name in enumerate(classes):
                        item_list.append(ClassItem(i + 1, name))
                else:
                    item_list = classes
            result = ClassMap(item_list)
        else:
            raise Exception('Cannot convert type {} to ClassMap'.format(
                type(classes)))

        return result
