class PokemonMemory:
    def __init__(self, pyboy):
        self.pyboy = pyboy
        
        # Memory addresses
        self.LEVEL_ADDRESS = 0xD18C
        self.HP_ADDRESS = 0xD16C
        self.X_POS_ADDRESS = 0xD362
        self.Y_POS_ADDRESS = 0xD361
        self.MAP_N_ADDRESS = 0xD35E
        self.BADGE_COUNT_ADDRESS = 0xD356
        self.PARTY_ADDRESS = 0xD163
        self.RT_IN_CURRENT_MAP_POKEBALL_OAK_LAB = 0xD5AB
    
    def read_level(self):
        """Read current level"""
        return self.pyboy.memory[self.LEVEL_ADDRESS]

    def read_party(self):
        """Read current party"""
        return self.pyboy.memory[self.PARTY_ADDRESS]
    
    def read_hp(self):
        """Read current HP"""
        hp = self.pyboy.memory[self.HP_ADDRESS]
        max_hp = self.pyboy.memory[self.HP_ADDRESS + 1]
        return (hp, max_hp)
    
    def read_position(self):
        """Read player's current position"""
        x = self.pyboy.memory[self.X_POS_ADDRESS]
        y = self.pyboy.memory[self.Y_POS_ADDRESS]
        map_id = self.pyboy.memory[self.MAP_N_ADDRESS]
        return (x, y, map_id)
    
    def read_map(self):
        """Read current map ID"""
        return self.pyboy.memory[self.MAP_N_ADDRESS]
    
    def read_badges(self):
        """Read the number of badges"""
        return self.pyboy.memory[self.BADGE_COUNT_ADDRESS]

    def relevant_thing_in_current_map(self):
        """Read the number of relevant things in the current map"""
        if self.pyboy.memory[self.RT_IN_CURRENT_MAP_POKEBALL_OAK_LAB] == 1:
            return 1
        else:
            return 0

