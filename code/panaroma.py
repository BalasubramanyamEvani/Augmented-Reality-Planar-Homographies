# Import necessary functions
import numpy as np
import cv2
from cpselect.cpselect import cpselect
from opts import get_opts
from planarH import computeH_ransac


# Q4
def parse(objs):
    x1 = np.zeros((len(objs), 2))
    x2 = np.zeros((len(objs), 2))

    for i, obj in enumerate(objs):
        x1[i, :] = np.array([
            obj["img1_x"], obj["img1_y"]
        ])
        x2[i, :] = np.array([
            obj["img2_x"], obj["img2_y"]
        ])

    return x1, x2


def warp(im1_path, im2_path, opts):
    im1 = cv2.imread(im1_path)
    im2 = cv2.imread(im2_path)
    # objs = cpselect(im1_path, im2_path)
    # print(objs)
    
    # cmu
    # objs = [{'point_id': 1, 'img1_x': 1622.6784279977046, 'img1_y': 1638.734816818293, 'img2_x': 374.8462830915496, 'img2_y': 1485.4686910908936}, {'point_id': 2, 'img1_x': 2956.9952872715344, 'img1_y': 1386.2964920908116, 'img2_x': 1871.4449225473318, 'img2_y': 1431.374764363576}, {'point_id': 3, 'img1_x': 1541.5375379067286, 'img1_y': 1007.6390049995898, 'img2_x': 356.8149741824436, 'img2_y': 746.1850258175555}, {'point_id': 4, 'img1_x': 1992.3202606343739, 'img1_y': 2188.68973854602, 'img2_x': 753.5037701827714, 'img2_y': 2152.6271207278087}]
    
    # craig
    objs = [{'point_id': 1, 'img1_x': 2794.713507089582, 'img1_y': 998.6233505450368, 'img2_x': 1646.0535611835094, 'img2_y': 989.6076960904838}, {'point_id': 2, 'img1_x': 1821.0228259978685, 'img1_y': 2188.68973854602, 'img2_x': 501.0654454552896, 'img2_y': 2179.6740840914676}, {'point_id': 3, 'img1_x': 2695.5413080895, 'img1_y': 2215.736701909679, 'img2_x': 1483.7717810015565, 'img2_y': 2170.6584296369147}, {'point_id': 4, 'img1_x': 3308.6058109990977, 'img1_y': 1918.2201049094333, 'img2_x': 2033.7267027292837, 'img2_y': 1873.1418326366688}]
    
    # for pano right left
    # objs = [
    #     {
    #         'point_id': 1, 
    #         'img1_x': 1424.8665847061711, 
    #         'img1_y': 260.36406851897414, 
    #         'img2_x': 1081.9980985165148, 
    #         'img2_y': 309.6233505450375
    #     }, 
    #     {
    #         'point_id': 2, 
    #         'img1_x': 791.0638226374886, 
    #         'img1_y': 772.6606015900337, 
    #         'img2_x': 520.4422834193915, 
    #         'img2_y': 792.3643144004591
    #     }, 
    #     {
    #         'point_id': 3, 
    #         'img1_x': 942.1256208507498, 
    #         'img1_y': 559.2037128104255, 
    #         'img2_x': 668.2201294975821, 
    #         'img2_y': 582.1913777559218
    #     }, 
    #     {
    #         'point_id': 4, 
    #         'img1_x': 1339.4838291943279, 
    #         'img1_y': 641.3025161871979, 
    #         'img2_x': 1006.4671994098844, 
    #         'img2_y': 644.5864683222687
    #     }
    # ]
    
    x1, x2 = parse(objs)
    H, _ = computeH_ransac(x1, x2, opts)
    canvas = np.ones((im2.shape[0] * 2, im2.shape[1] * 2, 3), dtype=np.uint8) * 255
    for i in range(im1.shape[0]):
        for j in range(im1.shape[1]):
            canvas[i, j] = im1[i, j]
    mask = np.ones(im2.shape, dtype=np.uint8) * 255
    warped_mask = cv2.warpPerspective(
        mask, H, canvas.shape[:2]
    )
    warped_im2 = cv2.warpPerspective(
        im2, H, canvas.shape[:2]
    )
    indexes = np.where(warped_mask == 255)
    canvas[indexes] = warped_im2[indexes]
    cv2.imshow("warped", canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__=="__main__":
    opts = get_opts()
    # warp("../data/pano_left.jpg", "../data/pano_right.jpg", opts)
    warp("../data/craig_left.jpg", "../data/craig_right.jpg", opts)
    # warp("../data/cmu_left.jpg", "../data/cmu_right.jpg", opts)
    