import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.datasets import load_breast_cancer

class DecisionTreeClassifier:
  PLOT_STEP = 0.02

  def __init__(self, x_axis, y_axis, dataset):
    self.dataset = dataset
    self.attributes = dataset.data[:, [x_axis, y_axis]]
    self.target = dataset.target
    self.x_axis_label = dataset.feature_names[x_axis]
    self.y_axis_label = dataset.feature_names[y_axis]

  def minimum_attribute_value(self, attribute_index):
    return self.attributes[:, attribute_index].min() - 1

  def maximum_attribute_value(self, attribute_index):
    return self.attributes[:, attribute_index].max() + 1

  def train_classifier(self):
    return tree.DecisionTreeClassifier(
        criterion="entropy"
    ).fit(self.attributes, self.target)

  def evenly_distrubuted_atribute_values(self, attribute_index):
    return np.arange(
        self.minimum_attribute_value(attribute_index),
        self.maximum_attribute_value(attribute_index),
        self.PLOT_STEP
    )

  def determine_background_points(self):
    xx, yy = np.meshgrid(
        self.evenly_distrubuted_atribute_values(0),
        self.evenly_distrubuted_atribute_values(1)
    )
    return (xx, yy)

  def compute_bg_point_classifications(self, xx, yy, classifier):
    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    return Z

  def plot(self):
    classifier = self.train_classifier()
    xx, yy = self.determine_background_points()
    Z = self.compute_bg_point_classifications(xx, yy, classifier)
    cs = plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)
    plt.scatter(
        self.attributes[:, 0],
        self.attributes[:, 1],
        c=self.target.astype(np.float)
    )
    plt.xlabel(self.x_axis_label)
    plt.ylabel(self.y_axis_label)
    plt.show()


DecisionTreeClassifier(0, 1, load_breast_cancer()).plot()
