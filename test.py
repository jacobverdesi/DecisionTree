
class Tree(object):
    def __init__(self,left,right):
        self.left = left
        self.right = right
        self.name = None
        self.depth = None

    def __str__(self):
        return "\n"+'| ' * self.depth + self.name+" "+str(self.right)+str(self.left)


def recurse(name,depth):
    if depth > 2:
        leaf=Tree("","")
        leaf.name="Leaf"
        leaf.depth=3
        return None
    else:
        print(name,depth)
        L=recurse("L",depth+1)
        R=recurse("R",depth+1)
        tree=Tree(L,R)
        tree.depth=depth
        tree.name=name
        return tree
print(recurse("start",0))

