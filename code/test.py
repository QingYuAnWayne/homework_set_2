from skimage import io, filters, feature, img_as_float32
from skimage.transform import rescale
from skimage.color import rgb2gray
import helpers
import student

feature_width = 16
scale_factor = 0.5
image1_file = "../data/NotreDame/NotreDame1.jpg"
image1_color = img_as_float32(io.imread(image1_file))
image1 = rgb2gray(image1_color)
image2_file = "../data/NotreDame/NotreDame2.jpg"
image2_color = img_as_float32(io.imread(image2_file))
image2 = rgb2gray(image2_color)
x1, y1, x2, y2 = helpers.cheat_interest_points('/Users/qingyuan/Desktop/大三下/CV/homework_set_2/data/NotreDame/NotreDameEval.mat', scale_factor)
feature1 = student.get_features(image1, x1, y1, feature_width)
feature2 = student.get_features(image2, x2, y2, feature_width)
match, confidence = student.match_features(feature1, feature2)
print("test success!")
