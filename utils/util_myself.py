import cv2
import os
import pandas as pd
import numpy as np
from shapely.geometry import Polygon
from collections import namedtuple
from PIL import ImageFont, ImageDraw, Image

X = lambda x, y, a, b, c : a*x + b*y + c
Y = lambda x, y, d, e, f : d*x + e*y + f

def crop_image( input_path, image_name, save_path, output_width, output_height, overlap_length ) :
    img = cv2.imread( input_path + "/" + image_name )

    crop_num = 1
    width = img.shape[0]
    height = img.shape[1]
        
    now_x = 0
    while now_x < width :
      if now_x + output_width > width : now_x = width - output_width
      now_y = 0
      while now_y < height :
        if now_y + output_height > height : now_y = height - output_height

        cv2.imwrite( save_path + "/" + image_name[:-4] + "_" + str(crop_num) + ".jpg", img[ now_x:now_x+output_width, now_y:now_y+output_height ] )
        crop_num += 1

        if now_y != height - output_height : now_y = now_y + output_height - overlap_length
        else : now_y = height

      if now_x != width - output_width : now_x = now_x + output_width - overlap_length 
      else : now_x = width

def merge_txt_file(  input_path, image_name, save_path, image_size, overlap_length, original_path  ) :
    original_img_shape = cv2.imread( original_path + "/" + image_name ).shape
    merge_txt_num = len([ f for f in os.listdir( input_path ) if os.path.isfile(os.path.join( input_path, f )) and f.startswith( image_name[:-4] ) and f.endswith( '.txt' ) ])
    if os.path.exists( input_path + "/" + image_name[:-4] + ".txt") : merge_txt_num -= 1

    merge_file = []
    original_file = []
    for i in range( merge_txt_num ) :
        df = pd.DataFrame(columns=['top_left_x', 'top_left_y', 'top_right_x', 'top_right_y', 'bottom_right_x', 'bottom_right_y', 'bottom_left_x', 'bottom_left_y', 'width', 'height', 'score'])
        f = open( input_path + "/" + image_name[:-4] + "_" + str(i+1) + ".txt", 'r' )
        temp_all = []
        for line in f.readlines():
            temp_all.append(line.split(","))
            temp = line.split(",")[:8]
            temp = [ int(i) for i in temp ]
            temp = temp + [temp[2]-temp[0], temp[7]-temp[1]]
            temp = temp + [line.split(",")[-1]]
            df.loc[len(df)] = temp
        f.close()
        original_file.append( temp_all )
        merge_file.append( df )

    now_x = 0
    overlap_line_x = 0
    now_y = 0
    overlap_line_y = 0
    flag = False
    merge_label_file = []

    #blank_image = np.zeros((original_img_shape[0], original_img_shape[1],3), np.uint8)
    i_num = 0
    for i in merge_file :
      #print(now_x)
      for j in range( len(i) ) :

        if i.iloc[j]['bottom_right_x']+now_x >= overlap_line_x and i.iloc[j]['bottom_right_y']+now_y >= overlap_line_y  :
            #blank_image = cv2.rectangle( original_img, (i.iloc[j]['top_left_x']+now_x, i.iloc[j]['top_left_y']+now_y), (i.iloc[j]['bottom_right_x']+now_x, i.iloc[j]['bottom_right_y']+now_y), (0,0,255), 1)
             merge_label_file.append( [i.iloc[j]['top_left_x']+now_x, i.iloc[j]['top_left_y']+now_y, i.iloc[j]['top_right_x']+now_x, i.iloc[j]['top_right_y']+now_y, i.iloc[j]['bottom_right_x']+now_x, i.iloc[j]['bottom_right_y']+now_y, i.iloc[j]['bottom_left_x']+now_x, i.iloc[j]['bottom_left_y']+now_y, i.iloc[j]['score']] )


      now_x = now_x + image_size - overlap_length
      overlap_line_x = now_x + overlap_length

      if flag :
        flag = False
        now_x = 0
        overlap_line_x = 0
        now_y = now_y + image_size - overlap_length
        overlap_line_y = now_y + overlap_length

        if now_y + image_size > original_img_shape[0] :
          overlap_line_y = now_y + overlap_length
          now_y = original_img_shape[0] - image_size

      if now_x + image_size > original_img_shape[1] :
        overlap_line_x = now_x + overlap_length
        now_x = original_img_shape[1] - image_size
        flag = True
      i_num += 1
    
    f = open( save_path + "/" + image_name[:-4] + ".txt", "w" )
    for i in merge_label_file :
        i = [str(j) for j in i ]
        f.write( i[0] + "," + i[1] + "," + i[2] + "," + i[3] + "," + i[4] + "," + i[5] + "," + i[6] + "," + i[7] + "," + i[8] )
    f.close
    return merge_label_file

def xml_output( save_path, coordinate_convert, input_path, image_name, top_x, top_y, bottom_x, bottom_y, d_score, group_id, recognition_label, r_score ) :

    top_message = "<?xml version='1.0' encoding='ISO-8859-1'?>\n<?xml-stylesheet type='text/xsl' href='image_metadata_stylesheet.xsl'?>\n<dataset>\n   <name>dlib face detection dataset generated by ImgLab</name>\n   <comment>\n      This dataset is manually crafted or adjusted using ImgLab web tool\n      Check more detail on https://github.com/NaturalIntelligence/imglab\n   </comment>\n   <images>\n"
    bottom_message = "      </image>\n   </images>\n</dataset>\n"

    if coordinate_convert :
        convert_parameter = [] # [A,D,B,E,C,F]
        f = open( input_path + "/" + image_name[:-4] + ".jgw", "r" )
        for line in f.readlines():
            convert_parameter.append((float(line)))
        f.close

    f = open( save_path + "/" + image_name[:-4] + ".xml", "w", encoding='utf16' )
    f.write( top_message )
    f.write( "      <image file='{}'>\n".format( image_name ) )

    if coordinate_convert :
        for i in range( len( top_x ) ) :
            f.write( "         <box top='{}' left='{}' width='{}' height='{}' id='{}' x='{}' y='{}' score='{}'>\n            <label score='{}'>{}</label>\n            <correction></correction>\n         </box>\n".\
                     format( top_y[i], top_x[i], (bottom_x[i]-top_x[i]), (bottom_y[i]-top_y[i]), group_id[i],\
                             X(int((top_x[i]+bottom_x[i])/2), int((top_y[i]+bottom_y[i])/2), convert_parameter[0], convert_parameter[2], convert_parameter[4]),\
                             Y(int((top_x[i]+bottom_x[i])/2), int((top_y[i]+bottom_y[i])/2), convert_parameter[1], convert_parameter[3], convert_parameter[5]),\
                             d_score[i], 0.0, recognition_label[i] ))
    else :
        for i in range( len( top_x ) ) :
            f.write( "         <box top='{}' left='{}' width='{}' height='{}' id='{}\' x='{}' y='{}' score='{}'>\n            <label score='{}'>{}</label>\n            <correction></correction>\n         </box>\n".\
                     format( top_y[i], top_x[i], (bottom_x[i]-top_x[i]), (bottom_y[i]-top_y[i]), group_id[i],\
                             'None',\
                             'None',\
                             d_score[i], 0.0, recognition_label[i] ))

    f.write( bottom_message )
    f.close()

def drawbox( img, x1, y1, x2, y2, x3, y3, x4, y4, color, thickness, write_word, word, word_size = 15 ) :
  '''
   word
    1*---------*2    1*---------*2
     |         |      |         |
    4*---------*3    4*---------*3
                     word
  '''
  cv2.line( img, (x1, y1), (x2, y2), color, thickness )
  cv2.line( img, (x2, y2), (x3, y3), color, thickness )
  cv2.line( img, (x3, y3), (x4, y4), color, thickness )
  cv2.line( img, (x4, y4), (x1, y1), color, thickness )

  if write_word :
    fontpath = './colab_map_ocr/utils/simsun.ttc'
    font = ImageFont.truetype( fontpath, word_size )
    img_pil = Image.fromarray( img )
    draw = ImageDraw.Draw( img_pil )
    if y1 - word_size > 0 :
      bbox = draw.textbbox((x1, y1-word_size), word, font=font)
      draw.rectangle(bbox, fill=color)
      draw.text((x1, y1-word_size), word, font=font, fill=(0, 0, 0))
    else :
      bbox = draw.textbbox((x4, y4), word, font=font)
      draw.rectangle(bbox, fill=color)
      draw.text((x4, y4), word, font=font, fill=(0, 0, 0))
    img = np.array(img_pil)
  return img

def build_list_dict( x_max_list, x_min_list, y_max_list, y_min_list ) :
  list_ = []
  for i in range(len(x_max_list)) :
    list_.append( {
          'points': [(x_min_list[i], y_min_list[i]), (x_max_list[i], y_min_list[i]), (x_max_list[i], y_max_list[i]), (x_min_list[i], y_max_list[i])],
          'text': '',
          'ignore': False,} )
  return list_
  
def remove_overlap( x_max_list, x_min_list, y_max_list, y_min_list, recognition_list, merge_label ) :
  evaluator = DetectionIoUEvaluator( is_output_polygon = False, iou_constraint = 0.3 )
  cord = build_list_dict( x_max_list, x_min_list, y_max_list, y_min_list )
  result = evaluator.evaluate_image( cord, cord )
  overlap = [ [i['gt'],i['det']] for i in result['pairs'] if i['gt'] != i['det'] ]
  overlap = sorted( list(set([ i[0] if (x_max_list[i[0]]-x_min_list[i[0]])*(y_max_list[i[0]]-y_min_list[i[0]]) < (x_max_list[i[1]]-x_min_list[i[1]])*(y_max_list[i[1]]-y_min_list[i[1]]) else i[1] for i in overlap ]))) #???
  for j in range( len(overlap)-1, -1, -1 ) :
    del x_max_list[overlap[j]]
    del x_min_list[overlap[j]]
    del y_max_list[overlap[j]]
    del y_min_list[overlap[j]]
    del recognition_list[overlap[j]]
    del merge_label[overlap[j]]
  return x_max_list, x_min_list, y_max_list, y_min_list, recognition_list, merge_label
  
def remove_num_char( x_max_list, x_min_list, y_max_list, y_min_list, recognition_list, merge_label ) :
  for j in range( len(x_max_list)-1, -1, -1 ) :
    if ( recognition_list[j] >= '0' and recognition_list[j] <= '9') or ( recognition_list[j] >= 'A' and recognition_list[j] <= 'Z')  or ( recognition_list[j] >= 'a' and recognition_list[j] <= 'z') :
      del x_max_list[j]
      del x_min_list[j]
      del y_max_list[j]
      del y_min_list[j]
      del recognition_list[j]
      del merge_label[j]
  return x_max_list, x_min_list, y_max_list, y_min_list, recognition_list, merge_label

def iou_rotate(box_a, box_b, method='union'):
    rect_a = cv2.minAreaRect(box_a)
    rect_b = cv2.minAreaRect(box_b)
    r1 = cv2.rotatedRectangleIntersection(rect_a, rect_b)
    if r1[0] == 0:
        return 0
    else:
        inter_area = cv2.contourArea(r1[1])
        area_a = cv2.contourArea(box_a)
        area_b = cv2.contourArea(box_b)
        union_area = area_a + area_b - inter_area
        if union_area == 0 or inter_area == 0:
            return 0
        if method == 'union':
            iou = inter_area / union_area
        elif method == 'intersection':
            iou = inter_area / min(area_a, area_b)
        else:
            raise NotImplementedError
        return iou

class DetectionIoUEvaluator(object):
    def __init__(self, is_output_polygon=False, iou_constraint=0.5, area_precision_constraint=0.5):
        self.is_output_polygon = is_output_polygon
        self.iou_constraint = iou_constraint
        self.area_precision_constraint = area_precision_constraint

    def evaluate_image(self, gt, pred):

        def get_union(pD, pG):
            return Polygon(pD).union(Polygon(pG)).area

        def get_intersection_over_union(pD, pG):
            return get_intersection(pD, pG) / get_union(pD, pG)

        def get_intersection(pD, pG):
            return Polygon(pD).intersection(Polygon(pG)).area

        def compute_ap(confList, matchList, numGtCare):
            correct = 0
            AP = 0
            if len(confList) > 0:
                confList = np.array(confList)
                matchList = np.array(matchList)
                sorted_ind = np.argsort(-confList)
                confList = confList[sorted_ind]
                matchList = matchList[sorted_ind]
                for n in range(len(confList)):
                    match = matchList[n]
                    if match:
                        correct += 1
                        AP += float(correct) / (n + 1)

                if numGtCare > 0:
                    AP /= numGtCare

            return AP

        perSampleMetrics = {}

        matchedSum = 0

        Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')

        numGlobalCareGt = 0
        numGlobalCareDet = 0

        arrGlobalConfidences = []
        arrGlobalMatches = []

        recall = 0
        precision = 0
        hmean = 0

        detMatched = 0

        iouMat = np.empty([1, 1])

        gtPols = []
        detPols = []

        gtPolPoints = []
        detPolPoints = []

        # Array of Ground Truth Polygons' keys marked as don't Care
        gtDontCarePolsNum = []
        # Array of Detected Polygons' matched with a don't Care GT
        detDontCarePolsNum = []

        pairs = []
        detMatchedNums = []

        arrSampleConfidences = []
        arrSampleMatch = []

        evaluationLog = ""

        for n in range(len(gt)):
            points = gt[n]['points']
            # transcription = gt[n]['text']
            dontCare = gt[n]['ignore']

            if not Polygon(points).is_valid or not Polygon(points).is_simple:
                continue

            gtPol = points
            gtPols.append(gtPol)
            gtPolPoints.append(points)
            if dontCare:
                gtDontCarePolsNum.append(len(gtPols) - 1)

        evaluationLog += "GT polygons: " + str(len(gtPols)) + (" (" + str(len(
            gtDontCarePolsNum)) + " don't care)\n" if len(gtDontCarePolsNum) > 0 else "\n")

        for n in range(len(pred)):
            points = pred[n]['points']
            if not Polygon(points).is_valid or not Polygon(points).is_simple:
                continue

            detPol = points
            detPols.append(detPol)
            detPolPoints.append(points)
            if len(gtDontCarePolsNum) > 0:
                for dontCarePol in gtDontCarePolsNum:
                    dontCarePol = gtPols[dontCarePol]
                    intersected_area = get_intersection(dontCarePol, detPol)
                    pdDimensions = Polygon(detPol).area
                    precision = 0 if pdDimensions == 0 else intersected_area / pdDimensions
                    if (precision > self.area_precision_constraint):
                        detDontCarePolsNum.append(len(detPols) - 1)
                        break


        if len(gtPols) > 0 and len(detPols) > 0:
            # Calculate IoU and precision matrixs
            outputShape = [len(gtPols), len(detPols)]
            iouMat = np.empty(outputShape)
            gtRectMat = np.zeros(len(gtPols), np.int8)
            detRectMat = np.zeros(len(detPols), np.int8)
            if self.is_output_polygon:
                for gtNum in range(len(gtPols)):
                    for detNum in range(len(detPols)):
                        pG = gtPols[gtNum]
                        pD = detPols[detNum]
                        iouMat[gtNum, detNum] = get_intersection_over_union(pD, pG)
            else:
                # gtPols = np.float32(gtPols)
                # detPols = np.float32(detPols)
                for gtNum in range(len(gtPols)):
                    for detNum in range(len(detPols)):
                        pG = np.float32(gtPols[gtNum])
                        pD = np.float32(detPols[detNum])
                        iouMat[gtNum, detNum] = iou_rotate(pD, pG)
            for gtNum in range(len(gtPols)):
                for detNum in range(len(detPols)):
                    if gtNum not in gtDontCarePolsNum and detNum not in detDontCarePolsNum:
                        if iouMat[gtNum, detNum] > self.iou_constraint :
                            gtRectMat[gtNum] = 1
                            detRectMat[detNum] = 1
                            detMatched += 1
                            pairs.append({'gt': gtNum, 'det': detNum})
                            detMatchedNums.append(detNum)
                            evaluationLog += "Match GT #" + \
                                             str(gtNum) + " with Det #" + str(detNum) + "\n"

        perSampleMetrics = {
            'pairs': pairs
        }

        return perSampleMetrics