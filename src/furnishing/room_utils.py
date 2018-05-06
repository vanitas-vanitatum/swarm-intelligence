import pickle

from src.furnishing.furniture_collection import *
from src.furnishing.furniture_construction import *
from src.furnishing.room import Room


def load_example_positions_for_example_room(path):
    with open(path, 'rb') as f:
        return np.concatenate(pickle.load(f))


def get_example_room():
    room = Room(15, 15)
    room.add_furniture(RectangularFurniture(7, 3, 90, 3, 3, False, True))
    room.add_furniture(Cupboard(3, 12, 180, 2, 2, False, open_angle=90))

    sofa = Sofa(5.5, 7, 90, 4, 2, False)
    tv = Tv(10.5, 7, -90, 5, 2, 3, 1, False, name='tv')

    room.add_furniture(RoundedCornersFurniture(7, 13, 35, 1, 1, True, True))
    room.add_furniture(Window(15, 7, room.width, room.height, 4))
    room.add_furniture(sofa)
    room.add_furniture(Door(0, 7, room.width, room.height, 4))
    room.add_furniture(tv)
    room.update_carpet_diameter()
    return room
