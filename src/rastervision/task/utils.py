from typing import (List, Dict, Tuple, Union)

from rastervision.protos.class_item_pb2 import ClassItem as ClassItemMsg
from rastervision.core.class_map import (ClassItem, ClassMap)

# TODO: Unit test

def construct_class_map(classes: Union[ClassMap,
                                     List[str],
                                     List[ClassItemMsg],
                                     List[ClassItem],
                                     Dict[str, int],
                                     Dict[str, Tuple[int, str]]]) -> ClassMap:
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
            if type(list(classes.items())[0]) is tuple:
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
                    item_list.append(ClassItem(item.id, item.name, item.color))
            elif type(classes[0]) is str:
                for i, name in enumerate(classes):
                    item_list.append(ClassItem(i + 1, name))
            else:
                item_list = classes
        result = ClassMap(item_list)
    else:
         raise Exception("Cannot convert type {} to ClassMap".format(type(classes)))

    return result

def classes_to_class_items(class_map):
    """Transform a ClassMap into
       a list of ClassItem protobuf messages
    """
    return [ClassItemMsg(name=item.name, id=item.id, color=item.color)
            for item in class_map.get_items()]
