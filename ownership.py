class OwnerShip:
    own: int

    def __init__(self, own: int):
        self.own = own

    def add(self):
        self.own += 1

    def value(self) -> int:
        return self.own


to_own = 1

o_ship = OwnerShip(to_own)
o_ship.add()

print(f"now to_own is {to_own}")
print(f"now OwnerShip is {o_ship.own}")


class OwnerShipTwo:
    own: OwnerShip

    def __init__(self, own: OwnerShip):
        self.own = own

    def add(self):
        self.own.add()

    def value(self) -> int:
        return self.own.value()


o2_ship = OwnerShipTwo(o_ship)
o2_ship.add()
print(f"now OwnerShip is {o_ship.value()}")
print(f"now OwnerShip2 is {o2_ship.own.value()}")

def own_ship_fun(own: int):
    b = own
    b += 1

to_own_2 = 1
own_ship_fun(to_own_2)

print(f"to_own_2 :{to_own_2}")
