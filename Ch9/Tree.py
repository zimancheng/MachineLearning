class TreeNode(object):
    """Base class for linear models."""

    def __init__(self, feature, value, left=None, right=None):
        """
        Args:
            feature: split on the feature
            value: feature value to split on
            left: left branch, could be numeric values, vectors, or other trees 
            right: right branch, could be numeric values, vectors, or other trees 
        """
        self.feature = feature
        self.value = value
        self.left = left
        self.right = right
        

