from src.furnishing.furniture import EllipseFurniture, RoundedFurniture, RectangularFurniture
from src.furnishing.room import Room, RoomDrawer


room = Room(15, 15)
room.add_furniture(RectangularFurniture(7, 3, 0, 3, 3))
room.add_furniture(EllipseFurniture(7, 7, 0, 3, 1))
drawer = RoomDrawer(room)
drawer.draw_all()
