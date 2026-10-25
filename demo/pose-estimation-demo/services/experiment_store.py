class ExperimentStore:
    def __init__(self): self.items={}
    def put(self,key,value): self.items[key]=value
    def get(self,key): return self.items.get(key)
