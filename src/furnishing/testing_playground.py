from src.furnishing.furniture_collection import *
from src.furnishing.furniture_construction import *
from src.furnishing.room import Room, RoomDrawer

room = Room(15, 15)
room.add_furniture(RectangularFurniture(7, 3, 90, 3, 3, False, True))
room.add_furniture(Cupboard(7, 7, 45, 3, 2, False))

room.add_furniture(RoundedCornersFurniture(7, 13, 35, 1, 1, True, True))
room.add_furniture(Window(15, 7, room.width, room.height, 4))
room.add_furniture(Door(0, 7, room.width, room.height, 4))
room.add_furniture(Tv(13, 7, 270, 5, 3, 3, 1, True, name='tv'))
drawer = RoomDrawer(room)
drawer.draw_all(
    tv_tv='yellow'
)
