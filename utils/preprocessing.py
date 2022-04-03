import cv2
from pathlib import Path

# 分类器初始化

cascade_path = Path(__file__).absolute().parent / 'cat_face.xml'
cascade = cv2.CascadeClassifier(cascade_path.__str__())


def capture_cat_face(frame):
    frame = cv2.resize(frame, (512, 512))
    # 灰度处理
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 猫脸检测
    facerect = cascade.detectMultiScale(frame_gray, scaleFactor=1.1, minNeighbors=5, minSize=(5, 5))
    faces = []
    if len(facerect) > 0:
        for (i, (x, y, w, h)) in enumerate(facerect):
            # 在猫脸区域画出方框
            # cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), thickness=2)
            # 使用ROI获取猫脸区域图像
            roiImg = frame[y:y + h, x:x + w]
            faces.append(roiImg)
    return faces


def get_face_area_size(face):
    x, y, _ = face.shape
    return x * y


img_path = cascade_path.parent.parent / 'datasets/images_background'
new_img_path = cascade_path.parent.parent / 'datasets/images_background_dealed'
if not new_img_path.exists():
    new_img_path.mkdir()


for folder in img_path.glob("*"):
    new_folder = new_img_path / folder.name
    print("Dealing with {}".format(folder.name))
    if not new_folder.exists():
        new_folder.mkdir()
    for file in folder.glob('*'):
        images = capture_cat_face(cv2.imread(file.absolute().__str__()))
        if len(images) == 0:
            continue
        image = max(images, key=get_face_area_size)
        x, y, z = image.shape
        if x < 170 or y < 170:
            continue
        cv2.imwrite(f"{new_folder.absolute().__str__()}/{file.name}", image)
        # print(image.shape)
        # cv2.imshow("cat", image)
        # cv2.waitKey()
        # cv2.destroyAllWindows()
