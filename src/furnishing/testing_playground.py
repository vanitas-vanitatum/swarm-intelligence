from src.furnishing.furniture_collection import *
from src.furnishing.furniture_construction import EllipseFurniture, RectangularFurniture
from src.furnishing.room import Room, RoomDrawer

room = Room(15, 15)
room.add_furniture(RectangularFurniture(7, 3, 90, 3, 3, False))
room.add_furniture(EllipseFurniture(7, 7, 0, 3, 1, False))
room.add_furniture(Window(15, 7, room.width, room.height, 4))
room.add_furniture(Door(0, 7, room.width, room.height, 4))
drawer = RoomDrawer(room)
drawer.draw_all()
