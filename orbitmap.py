import os
import subprocess
import shlex
import cv2
import math
import mediapipe as mp
import numpy as np
from PIL import Image
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import matplotlib.axis as axis
import csv
from operator import itemgetter
import time

start = time.time()

os.chdir('/Users/georgienahass/Desktop/PurnellTranLab/orbitmap/mediapipe/')
cmd = shlex.split('bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 mediapipe/examples/desktop/iris_tracking:iris_depth_from_image_desktop')
subprocess.run(cmd)



photos = '/Users/georgienahass/Desktop/PurnellTranLab/ghasemTED/exifAddedTEDPhotos/'
# alvin = '/Users/georgienahass/Desktop/rotatedForTeam2/'

"Calculate midpoint of 2 points. Use in face mesh facial mapping"
def midpoint(p1, p2):
    mid = (int((p1[0]+p2[0])/2), int((p1[1]+p2[1])/2))
    return mid

"Add facemesh to image and optionally save"
def mapTesselations(root, image_path_save, cf,subj):
    try:
        img = cv2.imread(image_path_save)
        mpDraw = mp.solutions.drawing_utils
        mpFacemesh = mp.solutions.face_mesh
        faceMesh = mpFacemesh.FaceMesh(static_image_mode=True, refine_landmarks=True)
        drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=1)
        results = faceMesh.process(img)
        id_list = []
        if results.multi_face_landmarks:
            for landmark in results.multi_face_landmarks:
                mpDraw.draw_landmarks(img, landmark, mpFacemesh.FACEMESH_TESSELATION, drawSpec,drawSpec)
                for id,lm in enumerate(landmark.landmark):
                    [ih, iw, ic] = img.shape
                    px,py,pz =int(lm.x*iw), int(lm.y*ih), (lm.z*iw)
                    append = (px,py,pz)
                    id_list.append(append)   
        arr = np.array(id_list, np.int32)

        subnasale = id_list[2]
        pronsale = id_list[4]
        R_ectocathion = id_list[130]
        L_ectocathion = id_list[359]
        R_cheilion = id_list[61]
        L_cheilion = id_list[291]
        superior_labrale = id_list[0]
        inferior_labrale = id_list[17]

        r =2

        cf = 11.71/cf
        
        Recto_chel_cf = round(cf * abs(math.dist(R_ectocathion, R_cheilion))/10, r) 
        Lecto_chel_cf = round(cf * abs(math.dist(L_ectocathion, L_cheilion))/10, r)
        Recto_pro_cf = round(cf * abs(math.dist(R_ectocathion, pronsale))/10, r)
        Lecto_pro_cf = round(cf * abs(math.dist(L_ectocathion, pronsale))/10, r)
        subn_SL_cf = round(cf * abs(math.dist(subnasale, superior_labrale))/10, r)
        subn_IL_cf= round(cf * abs(math.dist(subnasale,inferior_labrale))/10, r)
        
        c = (255,0,0)
        t = 2
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1
        # cv2.line(img, R_ectocathion[0:2], R_cheilion[0:2], c, t) 
        # mid_Recto_cheil = midpoint(R_ectocathion, R_cheilion)
        # cv2.putText(img, str(Recto_chel_cf)+' cm', mid_Recto_cheil, font, fontScale, c, t)

        # cv2.line(img, L_ectocathion[0:2], L_cheilion[0:2], c, t) 
        # mid_Lecto_cheil = midpoint(L_ectocathion, L_cheilion)
        # cv2.putText(img, str(Lecto_chel_cf)+' cm', mid_Lecto_cheil, font, fontScale, c, t)

        # cv2.line(img, R_ectocathion[0:2], pronsale[0:2], c, t) 
        # mid_Recto_pro = midpoint(R_ectocathion, pronsale)
        # cv2.putText(img, str(Recto_pro_cf)+' cm', mid_Recto_pro, font, fontScale, c, t)

        # cv2.line(img, L_ectocathion[0:2], pronsale[0:2], c, t) 
        # mid_Lecto_pro = midpoint(L_ectocathion, pronsale)
        # cv2.putText(img, str(Lecto_pro_cf)+' cm' , mid_Lecto_pro, font, fontScale, c, t)

        # cv2.line(img, subnasale[0:2], superior_labrale[0:2], c, t) 
        # mid_sub_Slab = midpoint(subnasale, superior_labrale)
        # cv2.putText(img, str(subn_SL_cf)+' cm', mid_sub_Slab, font, fontScale, c, t)

        # cv2.line(img, subnasale[0:2], inferior_labrale[0:2], c, t) 
        # mid_sub_Ilab = midpoint(subnasale, inferior_labrale)
        # cv2.putText(img, str(subn_IL_cf)+' cm', mid_sub_Ilab, font, fontScale, c, t)
        isClosed=True
        
    except (RuntimeError, TypeError, NameError, KeyError, IndexError, FileNotFoundError):
        pass    
    return id_list, cf, img, Recto_chel_cf*10, Lecto_chel_cf*10, Recto_pro_cf*10, Lecto_pro_cf*10, subn_SL_cf*10, subn_IL_cf*10

def dot(vA, vB):
    return vA[0]*vB[0]+vA[1]*vB[1]

def ang(lineA, lineB):
    # Get nicer vector form
    vA = [(lineA[0][0]-lineA[1][0]), (lineA[0][1]-lineA[1][1])]
    vB = [(lineB[0][0]-lineB[1][0]), (lineB[0][1]-lineB[1][1])]
    # Get dot prod
    dot_prod = dot(vA, vB)
    # Get magnitudes
    magA = dot(vA, vA)**0.5
    magB = dot(vB, vB)**0.5
    # Get cosine value
    cos_ = dot_prod/magA/magB
    # Get angle in radians and then convert to degrees
    angle = math.acos(dot_prod/magB/magA)
    return(angle)

"get the shortest distance to a line defined by two points from https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line#Line_defined_by_two_points"
def distance_to_line(p1, p2, p0):
    x0,y0 = p0
    x1,y1=p1
    x2,y2 = p2
    num = abs(((x2-x1) * (y1-y0)) - ((x1-x0)*(y2-y1)))
    den = math.sqrt((x2-x1)**2 + (y2-y1)**2)
    return num / den

def rotate(origin, points, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = origin
    rotate_points = []
    for point in points:
        px, py = point

        qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
        qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
        temp = [qx,qy]
        rotate_points.append(temp)
    return rotate_points

def rotate_face(face_mesh_xyz, left_lower_lid, right_lower_lid, left_upper_lid, right_upper_lid, iris_points):
    #use point 168 (nasion) and 10 (straight line up from nose) to define angle
    point_1 = face_mesh_xyz[168]
    point_2 = face_mesh_xyz[10]

    #make new point to define y axis
    point_3 = [point_1[0], point_2[1]] 

    line_A = [point_1, point_2]
    line_B = [point_1, point_3]
    #calculate angle to rotate all the points by
    angle = ang(line_A, line_B)
    if point_1[0] < point_2[0]:
        angle = angle*-1

    ll = rotate(point_1, left_lower_lid, angle)
    rl = rotate(point_1, right_lower_lid, angle)
    lu = rotate(point_1, left_upper_lid, angle)
    ru = rotate(point_1, right_upper_lid, angle)
    fm = rotate(point_1, face_mesh_xyz, angle)
    ir = rotate(point_1, iris_points, angle)

    #return rotated points
    return ll, rl, lu, ru, fm, ir, angle

"isolate top regions of eye lids"
def get_regions_top(min_xy, max_xy, high_xy, out, data):
    for pair in data:
        if pair[1] <= min_xy[1] and pair[0] <= high_xy[0]:
            out.append(pair)
        if pair[1] <= max_xy[1] and pair[0] >= high_xy[0]:
            out.append(pair)
    out=np.vstack(out)
    return out
      
"Function to isolate bottom part of left and right eyes"
def get_regions_bottom(min_xy, max_xy, low_xy, out, data):
    for pair in data:
        if pair[1] >= min_xy[1] and pair[0] <= low_xy[0]:
            out.append(pair)
        if pair[1] >= max_xy[1] and pair[0] >= low_xy[0]:
            out.append(pair)
    out=np.vstack(out)
    return out

"predict best fit of lid points extracted from mediapipe iris"
def predictFit(x,y):
    x = x.reshape(-1,1)
    x_sorted = np.sort(x,axis=0)

    pre_process = PolynomialFeatures(degree=4)

    x_poly = pre_process.fit_transform(x)

    pr_model = LinearRegression()
    pr_model.fit(x_poly, y)

    y_pred = pr_model.predict(x_poly)
    y_pred_sorted = pr_model.predict(pre_process.fit_transform(x_sorted))
    feat_names = pre_process.get_feature_names_out()
    coef = pr_model.coef_

    intercept = pr_model.intercept_
    initial = intercept+coef[0]

    r2 = (r2_score(y, y_pred))
    # formula = f'y={initial} + {coef[1]}x + {coef[2]}x^2 + {coef[3]}x^3 + {coef[4]}x^4' 
    return y_pred_sorted,x_sorted, r2, initial, coef

'Very important function to extract lids and iris points from mediapipe output based on color tracings. Return xy coords of lids and iris for later stuff'
def findLids(path):
    im = cv2.imread(path)

    black = [0,0,0]
    right = (255,0,255)
    left = (255,255,0)

    #### Make mask of all right eye and get relevant values for mapping
    mask1 = np.all(im != right, axis=-1)
    # Make all not colored pixels black
    im[mask1] = black
    interest_right = [255,0,255]
    ry, rx = np.where(np.all(im==interest_right,axis=2))

    zipped_right = np.column_stack((rx,ry))

    r_x_min_index = (np.argmin(rx))
    r_x_max_index = (np.argmax(rx))
    #lowest point in y direction
    r_y_max_index = (np.argmax(ry))
    #highest point in y direction
    r_y_min_index = (np.argmin(ry))

    right_min_xy = zipped_right[r_x_min_index]
    right_max_xy = zipped_right[r_x_max_index]
    #lowest xy value
    right_low_xy = zipped_right[r_y_max_index]
    #highest xy value
    right_high_xy = zipped_right[r_y_min_index]

    ###### Make mask of all left eye and get relevant values for mapping
    mask2 = np.all(im != left, axis=-1)
    # Make all not colored pixels black
    im[mask2] = black
    interest_left = [255,255,0]
    ly, lx = np.where(np.all(im==interest_left,axis=2))
    zipped_left = np.column_stack((lx,ly))

    l_x_min_index = (np.argmin(lx))
    l_x_max_index = (np.argmax(lx))
    #lowest point in y direction
    l_y_max_index = (np.argmax(ly))
    #highest point in y direction
    l_y_min_index = (np.argmin(ly))

    left_min_xy = zipped_left[l_x_min_index]
    left_max_xy = zipped_left[l_x_max_index]
    #lowest point in y direction
    left_low_xy = zipped_left[l_y_max_index]
    #highest point in y direction
    left_high_xy = zipped_left[l_y_min_index]

    #define output lists --maybe could make np arrays byt it works
    top_L = []
    bottom_L = []
    top_R = []
    bottom_R = []

    #get top region of left eye
    top_L = get_regions_top(left_min_xy, left_max_xy, left_high_xy, top_L, zipped_left)  

    #get top region of right eye
    top_R = get_regions_top(right_min_xy, right_max_xy, right_high_xy,  top_R, zipped_right)  

    #get bottom region of left eye
    bottom_L = get_regions_bottom(left_min_xy, left_max_xy, left_low_xy, bottom_L, zipped_left)  

    # #get bottom region of right eye
    bottom_R = get_regions_bottom(right_min_xy, right_max_xy, right_low_xy, bottom_R, zipped_right)  

    pred_y1, x_sort1, r1, initial_tl, coef_tl = predictFit(top_L[:,0],top_L[:,1])
    pred_y2, x_sort2, r2, initial_bl, coef_bl =predictFit(bottom_L[:,0],bottom_L[:,1])
    pred_y3, x_sort3, r3, initial_tr, coef_tr =predictFit(top_R[:,0],top_R[:,1])
    pred_y4, x_sort4, r4, initial_br, coef_br =predictFit(bottom_R[:,0],bottom_R[:,1])

    iris = (0,255,255)

    mask3 = np.all(im != iris, axis=-1)
    
    # Make all not colored pixels black
    im[mask3] = black            
    iy, ix = np.where(np.all(im==iris,axis=2))
    
    r_values = [r1,r2,r3,r4]
    
    s = []
    for x in range(len(ix)):
        s.append(1)
    
    return x_sort1,pred_y1, x_sort2,pred_y2, x_sort3, pred_y3,x_sort4,pred_y4, list(ix), list(iy), r_values, s, initial_tl, initial_bl, initial_tr, initial_br, coef_tl, coef_bl, coef_tr, coef_br

"take as input xy points of both irises and split them into left and right"
def splitIrisPoints(iris_points): 
    min_val = min(iris_points, key=itemgetter(0))
    max_val = max(iris_points, key=itemgetter(0))
    mid_xy = midpoint(min_val, max_val)
    right_iris = [item for item in iris_points if item[0]<mid_xy[0]]
    left_iris = [item for item in iris_points if item[0]>mid_xy[0]]
    return right_iris, left_iris

"detect mrd of both eyes using irises and lids"         
def detectMRD(left_lid_upper, left_lid_lower, right_lid_upper, right_lid_lower, R_iris_points, L_iris_points, cf, img):
    left_lid_upper_list = []
    for x in left_lid_upper:
        temp = []
        for y in x:
            y = int(y)
            temp.append(y)
        left_lid_upper_list.append(temp)

    left_lid_lower_list = []
    for x in left_lid_lower:
        temp = []
        for y in x:
            y = int(y)
            temp.append(y)
        left_lid_lower_list.append(temp)

    right_lid_upper_list = []
    for x in right_lid_upper:
        temp = []
        for y in x:
            y = int(y)
            temp.append(y)
        right_lid_upper_list.append(temp)
     

    right_lid_lower_list = []
    for x in right_lid_lower:
        temp = []
        for y in x:
            y = int(y)
            temp.append(y)
        right_lid_lower_list.append(temp)  

    ### do left eye first
    min_val = min(L_iris_points, key=itemgetter(0))
    max_val = max(L_iris_points, key=itemgetter(0)) 
   
    # input()
    mrd = midpoint(min_val, max_val)
    #upper lid (mrd 1)
    diff_in_x_up =[]
    for item in left_lid_upper:
        diff = abs(item[0] - mrd[0])
  
        diff_in_x_up.append(diff)
    index_x_min_upper = np.argmin(diff_in_x_up)
    mrd_1_left = math.dist(mrd, left_lid_upper[index_x_min_upper])
    mrd_1_left_cf = mrd_1_left*cf

    #lower lid (mrd 1)
    diff_in_x_low =[]
    for item in left_lid_lower:
        diff = abs(item[0] - mrd[0])
        diff_in_x_low.append(diff)
    index_x_min_lower = np.argmin(diff_in_x_low)
    mrd_2_left = math.dist(mrd, left_lid_lower[index_x_min_lower])
    mrd_2_left_cf = mrd_2_left*cf

    ### do right eye second
    min_val = min(R_iris_points, key=itemgetter(0))
    max_val = max(R_iris_points, key=itemgetter(0))    
    mrd_right = midpoint(min_val, max_val)

    #upper lid (mrd 1)
    diff_in_x_up =[]
    for item in right_lid_upper:
        diff = abs(item[0] - mrd_right[0])
        diff_in_x_up.append(diff)
    index_x_min_upper_ru = np.argmin(diff_in_x_up)
    mrd_1_right = math.dist(mrd_right, right_lid_upper[index_x_min_upper_ru])
    mrd_1_right_cf = mrd_1_right*cf

    #lower lid (mrd 1)
    diff_in_x_low =[]
    for item in right_lid_lower:
        diff = abs(item[0] - mrd_right[0])
        diff_in_x_low.append(diff)
    index_x_min_lower_rl = np.argmin(diff_in_x_low)
    mrd_2_right = math.dist(mrd_right, right_lid_lower[index_x_min_lower_rl])
    mrd_2_right_cf = mrd_2_right * cf

    c = (255,255,255)
    t = 2
    c1 = (0,0,0)
    c2 = (0,255,0)
    c3 = (0,0,255)
    c4 = (0, 234, 255)
    c5 = (255,0,0)

    cv2.line(img, list(mrd_right), right_lid_upper_list[index_x_min_upper_ru], c5, t) 
    cv2.line(img, list(mrd_right), right_lid_lower_list[index_x_min_lower_rl], c4, t) 
    cv2.line(img, mrd, left_lid_lower_list[index_x_min_lower], c4, t) 
    cv2.line(img, mrd, left_lid_upper_list[index_x_min_upper], c5, t) 

    # cv2.imshow('mrd', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return mrd_1_left_cf, mrd_2_left_cf, mrd_1_right_cf, mrd_2_right_cf, mrd, mrd_right

"detect superior scelral show of both eyes using irises and lids"  
def detectSupScleralShow(L_iris_points, R_iris_points, left_lid_upper, right_lid_upper,cf):
    max_val_left_iris = min(L_iris_points, key=itemgetter(1))  
    max_val_right_iris = min(R_iris_points, key=itemgetter(1))  
    max_val_left_lid = min(left_lid_upper, key=itemgetter(1))  
    max_val_right_lid = min(left_lid_upper, key=itemgetter(1))  
    
    ##signs inverted here because y axis is inverted on images
    diff_in_left = []
    if max_val_left_iris[1] <= max_val_left_lid[1]:
        L_sss_cf = 0
    else:
        for item in left_lid_upper:
            dist = abs(max_val_left_iris[0] - item[0])
            diff_in_left.append(dist) 
        index_x_min = np.argmin(diff_in_left)
        L_sss = math.dist(max_val_left_iris, left_lid_upper[index_x_min])
        L_sss_cf = L_sss * cf


    diff_in_right = []
    if max_val_right_iris[1] <= max_val_right_lid[1]:
        R_sss_cf = 0
    else:
        for item in right_lid_upper:
            dist = abs(max_val_right_iris[0] - item[0])
            diff_in_right.append(dist) 
        index_x_min = np.argmin(diff_in_right)
        R_sss = math.dist(max_val_right_iris, right_lid_upper[index_x_min])
        R_sss_cf = R_sss * cf

    return L_sss_cf, R_sss_cf
    
"detect inferior scelral show of both eyes using irises and lids"  
def detectInfScleralShow(L_iris_points, R_iris_points, left_lid_lower, right_lid_lower,cf):
    max_val_left_iris = max(L_iris_points, key=itemgetter(1))  
    max_val_right_iris = max(R_iris_points, key=itemgetter(1))  
    max_val_left_lid = max(left_lid_lower, key=itemgetter(1))  
    max_val_right_lid = max(left_lid_lower, key=itemgetter(1))  
   
   ##signs inverted here because y axis is inverted on images
    diff_in_left = []
    if max_val_left_iris[1] >= max_val_left_lid[1]:
        L_iss_cf = 0
    else:
        for item in left_lid_lower:
            dist = abs(max_val_left_iris[0] - item[0])
            diff_in_left.append(dist) 
        index_x_min = np.argmin(diff_in_left)
        L_iss = math.dist(max_val_left_iris, left_lid_lower[index_x_min])
        L_iss_cf = L_iss * cf

    diff_in_right = []
    if max_val_right_iris[1] >= max_val_right_lid[1]:
        R_iss_cf = 0
    else:
        for item in right_lid_lower:
            dist = abs(max_val_right_iris[0] - item[0])
            diff_in_right.append(dist) 
        index_x_min = np.argmin(diff_in_right)
        R_iss = math.dist(max_val_right_iris, right_lid_lower[index_x_min])
        R_iss_cf = R_iss * cf

    
    return L_iss_cf, R_iss_cf

"detect canthal height of both eyes.  return medial and lateral values of both eyes " 
"independently and also return xy coords of averaged medial and lateral canthi for later use"
def detectCanthalHeight(L_iris, R_iris, ll, lu, rl, ru,cf,img):
    ##### do left eye first
    min_val_iris_L = min(L_iris, key=itemgetter(0))
    max_val_iris_L = max(L_iris, key=itemgetter(0)) 
    #find center of iris
    center_L = midpoint(min_val_iris_L, max_val_iris_L)
    #define a line using iris center by making new point with same y value
    new_point_1L = [center_L[0]*-5, center_L[1]]
    new_point_2L = [center_L[0]*5, center_L[1]]

    #calculate the medial canthus by finding the min x value in the left upper and lower lid and average 
    max_val_ll = min(ll, key=itemgetter(0)) 
    max_val_lu = min(lu, key=itemgetter(0)) 
    l_medial_canthus = [(max_val_ll[0] + max_val_lu[0])/2 , (max_val_ll[1] + max_val_lu[1])/2]
    #calculate the lateral canthus by finding the maximum x value in the left upper and lower lid and average 
    min_val_ll = max(ll, key=itemgetter(0)) 
    min_val_lu = max(lu, key=itemgetter(0)) 
    l_lateral_canthus = [(min_val_ll[0] + min_val_lu[0])/2 , (min_val_ll[1] + min_val_lu[1])/2]

    mch_L = distance_to_line(new_point_1L, new_point_2L, l_medial_canthus)
    lch_L = distance_to_line(new_point_1L, new_point_2L, l_lateral_canthus)


    ##### do right eye now
    min_val_iris_R = min(R_iris, key=itemgetter(0))
    max_val_iris_R = max(R_iris, key=itemgetter(0)) 
    #find center of iris
    center_R = midpoint(min_val_iris_R, max_val_iris_R)
    #define a line using iris center by making new point with same y value
    new_point_1R = [center_R[0]*-5, center_R[1]]
    new_point_2R = [center_R[0]*5, center_R[1]]

    #calculate the medial canthus by finding the maximum x value in the right upper and lower lid and average 
    max_val_rl = max(rl, key=itemgetter(0)) 
    max_val_ru = max(ru, key=itemgetter(0)) 
    r_medial_canthus = [(max_val_rl[0] + max_val_ru[0])/2 , (max_val_rl[1] + max_val_ru[1])/2]
    #calculate the lateral canthus by finding the minimum x value in the right upper and lower lid and average 
    min_val_rl = min(rl, key=itemgetter(0)) 
    min_val_ru = min(ru, key=itemgetter(0)) 
    r_lateral_canthus = [(min_val_rl[0] + min_val_ru[0])/2 , (min_val_rl[1] + min_val_ru[1])/2]

    mch_R = distance_to_line(new_point_1R, new_point_2R, r_medial_canthus)
    lch_R = distance_to_line(new_point_1R, new_point_2R, r_lateral_canthus)

    mch_L_cf = mch_L * cf
    lch_L_cf = lch_L * cf
    mch_R_cf = mch_R * cf
    lch_R_cf = lch_R * cf

    c = (255,255,255)
    c1 = (255,0,255)
    c2 = (0,0,0)
    l_lateral_canthus_float = [int(x) for x in l_lateral_canthus]
    l_medial_canthus_float = [int(x) for x in l_medial_canthus]
    r_lateral_canthus_float = [int(x) for x in r_lateral_canthus]
    r_medial_canthus_float = [int(x) for x in r_medial_canthus]
    
    t = 2
    cv2.line(img, new_point_1L, new_point_2L, c, t) 
    cv2.line(img, center_L, l_lateral_canthus_float, c1, t) 
    cv2.line(img, center_L, l_medial_canthus_float, c2, t) 
    cv2.line(img, new_point_1R, new_point_2R, c, t) 
    cv2.line(img, center_R, r_lateral_canthus_float, c1, t) 
    cv2.line(img, center_R, r_medial_canthus_float, c2, t) 


    # cv2.imshow('mrd', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return mch_L_cf, lch_L_cf, mch_R_cf, lch_R_cf, r_lateral_canthus, r_medial_canthus, l_lateral_canthus, l_medial_canthus

"detect canthal tilt using angle between various lines of interest"
def detectCanthalTilt(facemesh,r_lateral_canthus, r_medial_canthus, l_lateral_canthus, l_medial_canthus,cf, img):
    #### do left eye first
    ## define line from medial canthus to lateral canthus using two xy points
    lmc_llc = [l_medial_canthus, l_lateral_canthus]

    ## define line from medial canthus to midline (use apex of nose x point) using two xy points
    point_1 = facemesh[1]
    mid_lmc = [[point_1[0], l_medial_canthus[1]], l_medial_canthus]

    # use ang function to get angle of two lines in radians and then convert to degrees
    lct_rad = ang(lmc_llc, mid_lmc)
    
    lct_deg = math.degrees(lct_rad)
    print(lct_deg)
    #### do right eye now
    ## define line from medial canthus to lateral canthus using two xy points
    rmc_rlc = [r_medial_canthus, r_lateral_canthus]

    ## define line from medial canthus to midline (use apex of nose x point) using two xy points
    mid_rmc = [r_medial_canthus, [point_1[0], r_medial_canthus[1]]]

    # use ang function to get angle of two lines in radians and then convert to degrees
    rct_rad = ang(mid_rmc, rmc_rlc)
    rct_deg = math.degrees(rct_rad)
    print(rct_deg)
    #subtract to get acute angle
    rct_deg = 180-rct_deg
    print(rct_deg)
    c = (255,255,255)
    t = 2
    c1 = (0,0,0)
    c2 = (0,255,0)
    c3 = (0,0,255)
    c4 = (0, 165, 255)
    
    point_1 = [int(x) for x in l_medial_canthus]
    point_2 = [int(x) for x in l_lateral_canthus]
    point_3 = [int(x) for x in [point_1[0], l_medial_canthus[1]]]
    point_4 = [int(x) for x in r_medial_canthus]
    point_5 = [int(x) for x in r_lateral_canthus]
    point_6 = [int(x) for x in  [point_1[0], r_medial_canthus[1]]]    



    cv2.line(img, point_1, point_2, c4, t) 
    # cv2.line(img, point_1, point_3, c4, t) 
    cv2.line(img, point_4, point_5, c4, t) 
    # cv2.line(img, point_4, point_6, c4, t) 





    return lct_deg, rct_deg

"use lateral and medial canthi to detect vertical dystopia"
def detectVerticalDystopia(r_medial_canthus, l_medial_canthus, facemesh,cf,img):
    ### define xy coord of l medial canthus to midline
    vertical_point = facemesh[1]
    l_mc = [vertical_point[0], l_medial_canthus[1]]
    ### define xy coord of R medial canthus to midline
    r_mc = [vertical_point[0], r_medial_canthus[1]]

    vert_dystop = math.dist(l_mc, r_mc)
    vert_dystop_cf = vert_dystop * cf

    c = (255,255,255)
    t = 2
    c1 = (0,0,0)
    c2 = (0,255,0)
    c3 = (0,0,255)
    
    point_1 = [int(x) for x in facemesh[1]]
    point_2 = [int(x) for x in facemesh[10]]
    point_3 = [int(x) for x in l_mc]
    point_4 = [int(x) for x in r_mc]
    point_5 = [int(x) for x in l_medial_canthus]
    point_6 = [int(x) for x in r_medial_canthus]    


    cv2.line(img, point_1, point_2, c, t) 
    cv2.line(img, point_5, point_3, c3, t) 
    cv2.line(img, point_6, point_4, c3, t) 

    return vert_dystop_cf

def detectBrowHeight(medial_canthus, lateral_canthus, lat_nimbus, facemesh, brow, lat_brow_index, cf,img):
    # combine x value of lateral nimbus with y value of medial canthus to get point with which to do math 
    lateral_nimbus_math = [lat_nimbus[0], medial_canthus[1]]

    # combine x value of lateral canthus with y value of medial canthis to lateral canthus point with which to do math
    lateral_canthus_math = [lateral_canthus[0], medial_canthus[1]]

    # define variable of facemesh identified left eyebrow lateral end point to get point with which to do math
    eyebrow_lateral_end = [facemesh[lat_brow_index][0], medial_canthus[1]]

    # find shortest value in predicted fit of left brow with smallest difference in x (similar to mrd) and do calculation with it
    # medial canthus
    diff_medial_canthus = []
    for item in brow:
        diff = abs(item[0] - medial_canthus[0])
        diff_medial_canthus.append(diff)
    index_x_med_canthus = np.argmin(diff_medial_canthus)
    medCanth_brow = math.dist(medial_canthus, brow[index_x_med_canthus])
    medCanth_brow_cf = medCanth_brow * cf
     
    # limbus 
    diff_limbus = []
    for item in brow:
        diff = abs(item[0] - lateral_nimbus_math[0])
        diff_limbus.append(diff)
    index_x_limbus = np.argmin(diff_limbus)
    limbus_brow = math.dist(lateral_nimbus_math, brow[index_x_limbus])
    limbus_brow_cf = limbus_brow * cf

    # lateral canthus
    diff_lateral_canthus = []
    for item in brow:
        diff = abs(item[0] - lateral_canthus_math[0])
        diff_lateral_canthus.append(diff)
    index_x_lat_canthus = np.argmin(diff_lateral_canthus)
    latCanth_brow = math.dist(lateral_canthus_math, brow[index_x_lat_canthus])
    latCanth_brow_cf = latCanth_brow * cf

    # lateral brow endpoint
    diff_lateral_brow = []
    for item in brow:
        diff = abs(item[0] - eyebrow_lateral_end[0])
        diff_lateral_brow.append(diff)
    index_x_lateral_brow = np.argmin(diff_lateral_brow)
    lateralBrow_brow = math.dist(eyebrow_lateral_end, brow[index_x_lateral_brow])
    lateralBrow_brow_cf = lateralBrow_brow * cf


    c = (255,255,255)
    t = 2
    c1 = (0,0,0)
    c2 = (0,255,0)
    c3 = (0,0,255)
    
    point_1 = [int(x) for x in lateral_nimbus_math]
    point_2 = [int(x) for x in lateral_canthus_math]
    point_3 = [int(x) for x in eyebrow_lateral_end]
    point_4 = [int(x) for x in medial_canthus]
    point_5 = [int(x) for x in brow[index_x_med_canthus]]
    point_6 = [int(x) for x in  brow[index_x_limbus]]    
    point_7 = [int(x) for x in brow[index_x_lat_canthus]]
    point_8 = [int(x) for x in brow[index_x_lateral_brow]]    


    cv2.line(img, point_1, point_6, c, t) 
    cv2.line(img, point_2, point_7, c1, t) 
    cv2.line(img, point_3, point_8, c2, t) 
    cv2.line(img, point_4, point_5, c2, t) 

    # cv2.imshow('mrd', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


    return medCanth_brow_cf, limbus_brow_cf, latCanth_brow_cf, lateralBrow_brow_cf, img

def saveRotatedImagesElsewhere(image_path, angle, outpath, root, name):	
    list_root = root.split(os.sep)
    path_addition = list_root[-2:]
    path_addition = '/'.join(path_addition)
    image = Image.open(image_path)
    imageCopy = image.copy()
    angle = angle *-1
    angle_deg = math.degrees(angle)
    im_rotate = imageCopy.rotate(angle_deg, expand=True)
    dir_to_save = os.path.join(outpath, path_addition)
    to_save = os.path.join(outpath, path_addition, name)
    os.makedirs(dir_to_save)
    im_rotate.save(to_save)

def rotate_image(image, angle, fm):
    image_center = fm[1]
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

def interCanthalDistance(r_mch, l_mch, r_lch, l_lch, l_iris, r_iris, cf,img):
    icd = math.dist(r_mch, l_mch) * cf
    ipd = math.dist(l_iris, r_iris) * cf
    ocd = math.dist(l_lch, r_lch) * cf

    return icd, ipd, ocd

def calculateLids(x_vals, initial, coeff):
    ## get min and max x value of upper left lid to generate lots of points
    min_x = min(x_vals, key=itemgetter(0)) 
    max_x = max(x_vals, key=itemgetter(0)) 
    x_data = np.linspace(min_x[0],  max_x[0], 150)
    plot_data = [1 for i in range(len(x_data))]
    y_vals = initial + coeff[1]*x_data + coeff[2]*x_data**2 + coeff[3]*x_data**3 + coeff[4]*x_data**4
    lid_coords = list(zip(x_data, y_vals))
    formula = (f'{initial} + {coeff[1]}x + {coeff[2]}x^2 + {coeff[3]}x^3 + {coeff[4]}x^4')            
    return formula, x_data, y_vals, lid_coords, plot_data

def calculateBrows(x_vals, y_vals, iris):
    y_pred, x_pred, r, initial, coef = predictFit(x_vals, y_vals) 
    min_x = min(x_pred, key=itemgetter(0)) 
    max_x = max(x_pred, key=itemgetter(0)) 
    brow_x = np.linspace(min_x[0],  max_x[0], 100)
    brow_plot = [1 for i in range(len(brow_x))]
    brow_y = initial + coef[1]*brow_x + coef[2]*brow_x**2 + coef[3]*brow_x**3 + coef[4]*brow_x**4
    brow = list(zip(brow_x, brow_y))
    lat_nimbus = max(iris, key=itemgetter(0)) 
    brow_form = (f'{initial} + {coef[1]}x + {coef[2]}x^2 + {coef[3]}x^3 + {coef[4]}x^4')            
    return brow, lat_nimbus, brow_plot, brow_form, brow_x, brow_y

def detectVerticalPalpFissure(lower_lid, upper_lid, cf, img):
    upper_point = max(upper_lid, key=itemgetter(1))  
    lower_point = min(lower_lid, key=itemgetter(1))  
    v_palp_fissure = math.dist(upper_point, lower_point) * cf
    
    t = 2
    c = (255,255,255)
    point_1 = [int(x) for x in upper_point]
    point_2 = [int(x) for x in lower_point]
    cv2.line(img, point_1, point_2, c, t) 
    return (v_palp_fissure)
    
def detectHorizontalPalpFissure(m_canthus, l_canthus, cf,img):
    h_palp_fissure = math.dist(m_canthus, l_canthus) * cf
    t = 2
    c1 = (0,0,0)
    point_1 = [int(x) for x in m_canthus]
    point_2 = [int(x) for x in l_canthus]
    cv2.line(img, point_1, point_2, c1, t) 
    return h_palp_fissure
    

x_fin = []
y_fin = []

failed = 0
succeed = 0
measure_filename = '/Calcuface.csv'
with open(photos + measure_filename,'w') as fout:
    line = ['photo_id','L_MRD_1','L_MRD_2','R_MRD_1','R_MRD_2','L_LCH','L_MCH','R_MCH','R_LCH','AVE_MCH',  
        'AVE_LCH','L_SSS','L_ISS','R_SSS','R_ISS','L_medCanBH','L_latLimBH','L_latCan_BH','L_latBrow_BH', 
        'R_medCanBH','R_latLimBH','R_latCan_BH','R_latBrow_BH','L_Canth_Tilt','R_Canth_Tilt','Vert_Dyst',
        'R_ectocathion_chelion','L_ectocathion_chelion','R_ectocathion_pronsale','L_ectocathion_pronsale',
        'Subnasale_superiorLabrale','Subnasale_inferiorLabrale', 'left_upLid_form', 'left_lowLid_form', 'right_upLid_form', 
        'right_lowLid_form',  'left_upBrow_form', 'left_lowBrow_form', 'right_upBrow_form', 'right_lowLid_form', 'icd', 'ipd', 'ocd', 'l_vertical_palp_fissure', 'r_vertical_palp_fissure',
        'l_horizontal_palp_fissure', 'r_horizontal_palp_fissure']
    fout.write("%s\n" % ",".join(line))

"main loop of algorithm"
for root, dirs,files in os.walk(photos):
    for file in files:
        print(file)
        if file.endswith('.jpg') or file.endswith('.JPG') or file.endswith('.jpeg'):
            if 'irisDetect' and 'facemesh' and 'fullAnnotation' and 'periorbit' not in file:
                pre_period = file.split('.', 1)[0]
                image = (os.path.join(root, file))
                print(f'IMAGE TO BE ANALYZED IS:  {image}')
                image_file_path_list = image.split(os.sep)
                subject_info_csv = image_file_path_list[-3:]
                subject_info_csv = '/'.join(subject_info_csv)
                ###OUTPUT PATH FOR DATA
                save_folder = os.path.join(root,'output')

                os.system('mkdir '+ save_folder )

                os.system('GLOG_logtostderr=1 bazel-bin/mediapipe/examples/desktop/iris_tracking/iris_depth_from_image_desktop  \
                                      --input_image_path=' + str(os.path.join(root, file)) + ' --output_image_path=' + str(os.path.join(save_folder, pre_period))+'_irisDetect.jpg')
                ## if iris detection file exists make a path variable for it for later. Dont really need this in this versino but nice to have 
                try:
                    iris_out = str(pre_period)+'_irisDetect.jpg'
                    real_image_path = str(os.path.join(save_folder, iris_out))
                    iris_pic = Image.open(real_image_path)
                    succeed += 1
                    
                except (RuntimeError, TypeError, NameError, KeyError, IndexError, FileNotFoundError):
                    print('Bad file... sorry')
                    failed +=1
                    break

                
                ## extract upper and lower lids  and iris by color and store
                mod_image_path = str(os.path.join(save_folder, iris_out))
                left_upper_lid_x,left_upper_lid_y, left_lower_lid_x,left_lower_lid_y,right_upper_lid_x,right_upper_lid_y,right_lower_lid_x,right_lower_lid_y, ix, iy, r_values, s, \
                     initial_tl, initial_bl, initial_tr, initial_br, coef_tl, coef_bl, coef_tr, coef_br  = findLids(mod_image_path)    

                """
                 whole idea is to populate the line of best fit of the lids with a lot 
                 of points to improve measurement distance on the face. Could wrap these in functions but
                 would still be messy and works fine now

                """
                
                lu_form, left_upper_lid_x, left_upper_lid_y, left_upper_lid, l_u = calculateLids(left_upper_lid_x, initial_tl, coef_tl)
                ru_form, right_upper_lid_x, right_upper_lid_y, right_upper_lid, r_u = calculateLids(right_upper_lid_x, initial_tr, coef_tr)   
                ll_form, left_lower_lid_x, left_lower_lid_y, left_lower_lid, l_l = calculateLids(left_lower_lid_x, initial_bl, coef_bl)            
                rl_form, right_lower_lid_x, right_lower_lid_y, right_lower_lid, r_l = calculateLids(right_lower_lid_x, initial_br, coef_br)
                
                # ## get min and max x value of upper left lid to generate lots of points
                # left_upper_lid_min_x = min(left_upper_lid_x, key=itemgetter(0)) 
                # left_upper_lid_max_x = max(left_upper_lid_x, key=itemgetter(0)) 
                # left_upper_lid_x = np.linspace(left_upper_lid_min_x[0],  left_upper_lid_max_x[0], 100)
                # l_u = [1 for i in range(len(left_upper_lid_x))]
                # left_upper_lid_y = initial_tl + coef_tl[1]*left_upper_lid_x + coef_tl[2]*left_upper_lid_x**2 + coef_tl[3]*left_upper_lid_x**3 + coef_tl[4]*left_upper_lid_x**4
            
 
                # ## get min and max x value of upper right lid to generate lots of points
                # right_upper_lid_min_x = min(right_upper_lid_x, key=itemgetter(0)) 
                # right_upper_lid_max_x = max(right_upper_lid_x, key=itemgetter(0)) 
                # right_upper_lid_x = np.linspace(right_upper_lid_min_x[0],  right_upper_lid_max_x[0], 100)
                # r_u = [1 for i in range(len(right_upper_lid_x))]
                # right_upper_lid_y = initial_tr + coef_tr[1]*right_upper_lid_x + coef_tr[2]*right_upper_lid_x**2 + coef_tr[3]*right_upper_lid_x**3 + coef_tr[4]*right_upper_lid_x**4
            

                # ## get min and max x value of bottom left lid to generate lots of points
                # left_lower_lid_min_x = min(left_lower_lid_x, key=itemgetter(0)) 
                # left_lower_lid_max_x = max(left_lower_lid_x, key=itemgetter(0)) 
                # left_lower_lid_x = np.linspace(left_lower_lid_min_x[0],  left_lower_lid_max_x[0], 100)
                # l_l = [1 for i in range(len(left_lower_lid_x))]
                # left_lower_lid_y = initial_bl + coef_bl[1]*left_lower_lid_x + coef_bl[2]*left_lower_lid_x**2 + coef_bl[3]*left_lower_lid_x**3 + coef_bl[4]*left_lower_lid_x**4

                # ## get min and max x value of upper left lid to generate lots of points
                # right_lower_lid_min_x = min(right_lower_lid_x, key=itemgetter(0)) 
                # right_lower_lid_max_x = max(right_lower_lid_x, key=itemgetter(0)) 
                # right_lower_lid_x = np.linspace(right_lower_lid_min_x[0], right_lower_lid_max_x[0], 100)
                # r_l = [1 for i in range(len(right_lower_lid_x))]
                # right_lower_lid_y = initial_br + coef_br[1]*right_lower_lid_x + coef_br[2]*right_lower_lid_x**2 + coef_br[3]*right_lower_lid_x**3 + coef_br[4]*right_lower_lid_x**4

                # #equations of lids for storage
                # rl_form = (f'{initial_br} + {coef_br[1]}x + {coef_br[2]}x^2 + {coef_br[3]}x^3 + {coef_br[4]}x^4')            
                # ru_form = (f'{initial_tr} + {coef_tr[1]}x + {coef_tr[2]}x^2 + {coef_tr[3]}x^3 + {coef_tr[4]}x^4')            
                # ll_form = (f'{initial_bl} + {coef_bl[1]}x + {coef_bl[2]}x^2 + {coef_bl[3]}x^3 + {coef_bl[4]}x^4')            
                # lu_form = (f'{initial_tl} + {coef_tl[1]}x + {coef_tl[2]}x^2 + {coef_tl[3]}x^3 + {coef_tl[4]}x^4')            

                # list of xy pairs of the lids to use for calculations
                # left_lower_lid = list(zip(left_lower_lid_x, left_lower_lid_y))
                # left_upper_lid =  list(zip(left_upper_lid_x, left_upper_lid_y))
                # right_lower_lid = list(zip(right_lower_lid_x, right_lower_lid_y))
                # right_upper_lid = list(zip(right_upper_lid_x, right_upper_lid_y))
                
                iris = list(zip(ix,iy))

                ## open text file created by mediapipe c++ iris (GRN version) and read in diamter values -> compute average
                diameter = []
                with open('iris_out.txt') as f:
                    for line in f:
                        diameter.append(line)
                for item in diameter:
                    item = item[:-2]
                diameter = [float(x[:-2]) for x in diameter]
                average = sum(diameter)/ len(diameter)
                
                ## delete txt file before next loop
                "NEED TO MAKE THIS SO IT INCLUDES ANY FILE WITH IRIS OUT IN IT IN CASE LEFTOVER"
                os.system('rm iris_out.txt')
                
                ## send ORIGINAL data to facemesh function. include: Conversion factor, coords of lid tracings, and iris tracings
                mesh, cf, img, mid_Recto_cheil, mid_Lecto_cheil, mid_Recto_pro, mid_Lecto_pro, mid_sub_Slab, mid_sub_Ilab = mapTesselations(root, image, average, pre_period)
                facemesh_path = os.path.join(save_folder,str(pre_period)+'_facemesh.jpg')
                cv2.imwrite(facemesh_path, img)
                mesh_x = [x[0] for x in mesh]
                mesh_y = [x[1] for x in mesh]
                mesh_no_z = [[x[0], x[1]] for x in mesh]
                im = cv2.imread(mod_image_path)

                #rotate face by angle calculated earlier  
                ll, rl, lu, ru, fm, ir, angle = rotate_face(mesh_no_z, left_lower_lid, right_lower_lid, left_upper_lid, right_upper_lid, iris)
                
                #make a copy of the image and rotate it to save for image j analysis
                tempImg = cv2.imread(image)
                degAng = math.degrees(angle)
                degAng = degAng*-1
                tempRotateImage = rotate_image(tempImg, degAng, fm)

                #make lists of rotated xy points for plotting 
                fmx = [x[0] for x in fm]
                fmy = [x[1] for x in fm]
                irx = [x[0] for x in ir]
                iry = [x[1] for x in ir]  

                #split iris points into left and right face
                right_iris, left_iris = splitIrisPoints(ir)

                #make a size list for scatter plot
                d = [1 for i in range(len(mesh_x))]

                "do the bulk of the facial detections"
                mrd_1_left, mrd_2_left, mrd_1_right, mrd_2_right,  iris_center_left, iris_center_right = detectMRD(lu, ll, ru, rl, right_iris, left_iris, cf, tempRotateImage)
                l_SSS, r_SSS = detectSupScleralShow(left_iris, right_iris, lu, ru, cf)
                l_ISS, r_ISS = detectInfScleralShow(left_iris, right_iris, ll, rl, cf)
                mch_L, lch_L, mch_R, lch_R, r_lateral_canthus, r_medial_canthus, l_lateral_canthus, l_medial_canthus = detectCanthalHeight(left_iris, right_iris, ll, lu, rl, ru, cf, tempRotateImage)
                lct_deg, rct_deg = detectCanthalTilt(fm,r_lateral_canthus, r_medial_canthus, l_lateral_canthus, l_medial_canthus,cf, tempRotateImage)
                vert_dystop = detectVerticalDystopia(r_medial_canthus, l_medial_canthus, fm,cf, tempRotateImage)
                icd, ipd, ocd = interCanthalDistance(r_medial_canthus, r_lateral_canthus, l_medial_canthus, l_lateral_canthus, iris_center_left, iris_center_right, cf, tempRotateImage)
                l_v_palp_fissure = detectVerticalPalpFissure(ll, lu, cf, image)
                r_v_palp_fissure = detectVerticalPalpFissure(rl, ru, cf, image)
                l_h_palp_fissure = detectHorizontalPalpFissure(l_medial_canthus, l_lateral_canthus, cf,image)
                r_h_palp_fissure = detectHorizontalPalpFissure(r_medial_canthus, r_lateral_canthus, cf, image)
                
                "calculate 4th degree polynomial of upper eyebrows using mediapipe and then use formula from predicted fit to fill in a bunch of points for brow height measurements"
                # calculate 4th degree polynomial of upper left eyebrow facemesh points using predict fit function
                # mediapipe points of upper left brow are [383, 300, 293, 334, 296, 336, 285, 417]
                left_upper_brow_x_np = np.array([fm[383][0], fm[300][0], fm[293][0], fm[334][0], fm[296][0], fm[336][0], fm[285][0], fm[417][0]])
                left_upper_brow_y_np = np.array([fm[383][1], fm[300][1], fm[293][1], fm[334][1], fm[296][1], fm[336][1], fm[285][1], fm[417][1]])
                # lu_y_pred, lu_x_pred, r, initial_lu, coef_lu = predictFit(left_upper_brow_x_np, left_upper_brow_y_np) 


                left_brow, lat_nimbus_left, l2, lu_brow_form, left_upper_brow_x, left_upper_brow_y = calculateBrows(left_upper_brow_x_np, left_upper_brow_y_np, left_iris)     
                L_medCanth_brow_cf, L_limbus_brow_cf, L_latCanth_brow_cf, L_lateralBrow_brow_cf, brow_1 = detectBrowHeight(l_medial_canthus, l_lateral_canthus, lat_nimbus_left, fm, left_brow, 383, cf, image)
                

                # #get min and max x value of upper left brow to generate lots of points
                # left_upper_brow_min_x = min(lu_x_pred, key=itemgetter(0)) 
                # left_upper_brow_max_x = max(lu_x_pred, key=itemgetter(0)) 
                # left_upper_brow_x = np.linspace(left_upper_brow_min_x[0],  left_upper_brow_max_x[0], 50)
                # l2 = [1 for i in range(len(left_upper_brow_x))]
                # left_upper_brow_y = initial_lu + coef_lu[1]*left_upper_brow_x + coef_lu[2]*left_upper_brow_x**2 + coef_lu[3]*left_upper_brow_x**3 + coef_lu[4]*left_upper_brow_x**4
                # left_brow = list(zip(left_upper_brow_x, left_upper_brow_y))
                # lat_nimbus_left = max(left_iris, key=itemgetter(0)) 
                # L_medCanth_brow_cf, L_limbus_brow_cf, L_latCanth_brow_cf, L_lateralBrow_brow_cf, brow_1 = detectBrowHeight(l_medial_canthus, l_lateral_canthus, lat_nimbus_left, fm, left_brow, 383, cf, tempRotateImage)
                
                # calculate 4th degree polynomial of upper right eyebrow facemesh points using predict fit function
                # mediapipe points of upper right brow are [156, 70, 63, 105, 66, 107, 55, 193],
                right_upper_brow_x_np = np.array([fm[156][0], fm[70][0], fm[63][0], fm[105][0], fm[66][0], fm[107][0], fm[55][0], fm[193][0]])
                right_upper_brow_y_np = np.array([fm[156][1], fm[70][1], fm[63][1], fm[105][1], fm[66][1], fm[107][1], fm[55][1], fm[193][1]])
                # ru_y_pred, ru_x_pred, r, initial_ru, coef_ru = predictFit(right_upper_brow_x, right_upper_brow_y)

                right_brow, lat_nimbus_right, r2, ru_brow_form, right_upper_brow_x, right_upper_brow_y = calculateBrows(right_upper_brow_x_np, right_upper_brow_y_np, right_iris)
                R_medCanth_brow_cf, R_limbus_brow_cf, R_latCanth_brow_cf, R_lateralBrow_brow_cf,brow_2 = detectBrowHeight(r_medial_canthus, r_lateral_canthus, lat_nimbus_right, fm, right_brow, 156, cf, image) 


                # #get min and max x value of upper right brow to generate lots of points
                # right_upper_brow_min_x = min(ru_x_pred, key=itemgetter(0)) 
                # right_upper_brow_max_x = max(ru_x_pred, key=itemgetter(0)) 
                # right_upper_brow_x = np.linspace(right_upper_brow_min_x[0],  right_upper_brow_max_x[0], 50)
                # r2 = [1 for i in range(len(right_upper_brow_x))]
                # right_upper_brow_y = initial_ru + coef_ru[1]*right_upper_brow_x + coef_ru[2]*right_upper_brow_x**2 + coef_ru[3]*right_upper_brow_x**3 + coef_ru[4]*right_upper_brow_x**4
                # right_brow = list(zip(right_upper_brow_x, right_upper_brow_y))
                # lat_nimbus_right = min(right_iris, key=itemgetter(0)) 
                # R_medCanth_brow_cf, R_limbus_brow_cf, R_latCanth_brow_cf, R_lateralBrow_brow_cf,brow_2 = detectBrowHeight(r_medial_canthus, r_lateral_canthus, lat_nimbus_right, fm, right_brow, 156, cf, tempRotateImage)                
                # periorbit_path = os.path.join(save_folder,str(pre_period)+'_periorbitMeasure.jpg')
                # cv2.imwrite(periorbit_path, brow_2)

                " make tracing of lower eyebrows just to have "
                # calculate 4th degree polynomial of lower left eyebrow facemesh points using predict fit function
                # mediapipe points of lower left brow are [265, 353, 276, 283, 282, 295]
                left_lower_brow_x = np.array([fm[265][0], fm[353][0], fm[276][0], fm[283][0], fm[282][0], fm[295][0]])
                left_lower_brow_y = np.array([fm[265][1], fm[353][1], fm[276][1], fm[283][1], fm[282][1], fm[295][1]])
                ll_y_pred, ll_x_pred, r, initial_ll, coef_ll = predictFit(left_lower_brow_x, left_lower_brow_y)
                l1 = [1 for i in range(len(ll_y_pred))]
     

                # calculate 4th degree polynomial of lower right eyebrow facemesh points using predict fit function
                # mediapipe points of lower right brow are [35, 124, 46, 53, 52, 65]
                right_lower_brow_x = np.array([fm[35][0], fm[124][0], fm[46][0], fm[53][0], fm[52][0], fm[65][0]])
                right_lower_brow_y = np.array([fm[35][1], fm[124][1], fm[46][1], fm[53][1], fm[52][1], fm[65][1]])
                rl_y_pred, rl_x_pred, r, initial_rl, coef_rl = predictFit(right_lower_brow_x, right_lower_brow_y)
                r1 = [1 for i in range(len(rl_y_pred))]



                #get equations of brows to write to spreadsheet
                # lu_brow_form = (f'{initial_lu} + {coef_lu[1]}x + {coef_lu[2]}x^2 + {coef_lu[3]}x^3 + {coef_lu[4]}x^4')            
                ll_brow_form = (f'{initial_ll} + {coef_ll[1]}x + {coef_ll[2]}x^2 + {coef_ll[3]}x^3 + {coef_ll[4]}x^4')            
                # ru_brow_form = (f'{initial_ru} + {coef_ru[1]}x + {coef_ru[2]}x^2 + {coef_ru[3]}x^3 + {coef_ru[4]}x^4')            
                rl_brow_form = (f'{initial_rl} + {coef_rl[1]}x + {coef_rl[2]}x^2 + {coef_rl[3]}x^3 + {coef_rl[4]}x^4')            

                #save rotated image for alvin and image j team
                # rot_image = saveRotatedImagesElsewhere(image, angle, alvin, root, file)

                plt.scatter(left_upper_lid_x,left_upper_lid_y,l_u)
                plt.scatter(left_lower_lid_x,left_lower_lid_y,l_l)
                plt.scatter(right_upper_lid_x,right_upper_lid_y,r_u)
                plt.scatter(right_lower_lid_x,right_lower_lid_y,r_l)
                plt.scatter(fmx, fmy, d, c='white')
                plt.scatter(irx,iry, s)
                plt.scatter(left_upper_brow_x, left_upper_brow_y,l2)
                plt.scatter(right_upper_brow_x, right_upper_brow_y,r2)
                plt.scatter(ll_x_pred, ll_y_pred, l1)
                plt.scatter(rl_x_pred, rl_y_pred, r1)

                plt.axis('off')
                tempRotateImage = cv2.cvtColor(tempRotateImage, cv2.COLOR_BGR2RGB)             
                plt.imshow(tempRotateImage)
                matplot_out = os.path.join(save_folder,str(pre_period)+'_fullAnnotation.jpg')
                plt.savefig(matplot_out)
                plt.close()
                # plt.show()

                ave_mch = (mch_R + mch_L) / 2
                ave_lch = (lch_R + lch_L) /2
                with open(photos + measure_filename,'a') as fout:
                    line = [subject_info_csv,mrd_1_left,mrd_2_left,mrd_1_right,mrd_2_right,lch_L,mch_L,mch_R,lch_R,ave_mch,  
                        ave_lch,l_SSS,l_ISS,r_SSS,r_ISS,L_medCanth_brow_cf,L_limbus_brow_cf,L_latCanth_brow_cf,L_lateralBrow_brow_cf, 
                        R_medCanth_brow_cf,R_limbus_brow_cf,R_latCanth_brow_cf,R_lateralBrow_brow_cf, lct_deg,rct_deg,vert_dystop,
                        mid_Recto_cheil, mid_Lecto_cheil, mid_Recto_pro, mid_Lecto_pro, mid_sub_Slab, mid_sub_Ilab,lu_form, 
                        ll_form, ru_form, rl_form,  lu_brow_form, ll_brow_form, ru_brow_form, rl_brow_form, icd, ipd, ocd, l_v_palp_fissure,
                        r_v_palp_fissure, l_h_palp_fissure, r_h_palp_fissure]
                    line = [str(l) for l in line]
                    fout.write("%s\n" % ",".join(line))



print(f'{failed} pictures failed to run')
print(f'{succeed} pictures succeeded')
print('The run is now complete')


finish = time.time()


time_tot = finish-start
print(time_tot)