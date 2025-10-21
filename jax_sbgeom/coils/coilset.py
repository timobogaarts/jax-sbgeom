from .base_coil import Coil
class CoilSet:
    '''
    Set of coils
    '''
    coils : list[Coil]

    def __init__(self, coils : list[Coil]):
        self.coils = coils

    def __getitem__(self, index : int) -> Coil:
        return self.coils[index]
    def __len__(self) -> int:
        return len(self.coils)
    