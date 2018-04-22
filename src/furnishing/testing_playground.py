from src.furnishing.furniture_collection import *
from src.furnishing.furniture_construction import *
from src.furnishing.room import Room, RoomDrawer

room = Room(15, 15)
room.add_furniture(RectangularFurniture(7, 3, 90, 3, 3, False, True))
room.add_furniture(Cupboard(4, 14, 180, 3, 2, False))

sofa = Sofa(7, 7, 90, 4, 2, True)
tv = Tv(10.5, 7, -90, 5, 2, 3, 1, True, name='tv')

room.add_furniture(RoundedCornersFurniture(7, 13, 35, 1, 1, True, True))
room.add_furniture(Window(15, 7, room.width, room.height, 4))
room.add_furniture(sofa)
room.add_furniture(Door(0, 7, room.width, room.height, 4))
room.add_furniture(tv)
room.update_carpet_diameter()
drawer = RoomDrawer(room)
drawer.draw_all(
    tv_tv='yellow'
)

print(sofa.can_see_tv(tv))
print(room.are_furniture_ok())
