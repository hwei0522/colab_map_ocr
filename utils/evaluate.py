from collections import namedtuple
import numpy as np
from shapely.geometry import Polygon
import cv2
from PIL import ImageFont, ImageDraw, Image
import argparse
import os
from tqdm import tqdm


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

        evaluationLog += "DET polygons: " + str(len(detPols)) + (" (" + str(len(
            detDontCarePolsNum)) + " don't care)\n" if len(detDontCarePolsNum) > 0 else "\n")

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
                    if gtRectMat[gtNum] == 0 and detRectMat[
                        detNum] == 0 and gtNum not in gtDontCarePolsNum and detNum not in detDontCarePolsNum:
                        if iouMat[gtNum, detNum] > self.iou_constraint:
                            gtRectMat[gtNum] = 1
                            detRectMat[detNum] = 1
                            detMatched += 1
                            pairs.append({'gt': gtNum, 'det': detNum})
                            detMatchedNums.append(detNum)
                            evaluationLog += "Match GT #" + \
                                             str(gtNum) + " with Det #" + str(detNum) + "\n"

        numGtCare = (len(gtPols) - len(gtDontCarePolsNum))
        numDetCare = (len(detPols) - len(detDontCarePolsNum))
        if numGtCare == 0:
            recall = float(1)
            precision = float(0) if numDetCare > 0 else float(1)
        else:
            recall = float(detMatched) / numGtCare
            precision = 0 if numDetCare == 0 else float(
                detMatched) / numDetCare

        hmean = 0 if (precision + recall) == 0 else 2.0 * \
                                                    precision * recall / (precision + recall)

        matchedSum += detMatched
        numGlobalCareGt += numGtCare
        numGlobalCareDet += numDetCare

        perSampleMetrics = {
            'precision': precision,
            'recall': recall,
            'hmean': hmean,
            'pairs': pairs,
            'iouMat': [] if len(detPols) > 100 else iouMat.tolist(),
            'gtPolPoints': gtPolPoints,
            'detPolPoints': detPolPoints,
            'gtCare': numGtCare,
            'detCare': numDetCare,
            'gtDontCare': gtDontCarePolsNum,
            'detDontCare': detDontCarePolsNum,
            'detMatched': detMatched,
            'evaluationLog': evaluationLog
        }

        return perSampleMetrics

    def combine_results(self, results):
        numGlobalCareGt = 0
        numGlobalCareDet = 0
        matchedSum = 0
        for result in results:
            numGlobalCareGt += result['gtCare']
            numGlobalCareDet += result['detCare']
            matchedSum += result['detMatched']

        methodRecall = 0 if numGlobalCareGt == 0 else float(
            matchedSum) / numGlobalCareGt
        methodPrecision = 0 if numGlobalCareDet == 0 else float(
            matchedSum) / numGlobalCareDet
        methodHmean = 0 if methodRecall + methodPrecision == 0 else 2 * \
                                                                            methodRecall * methodPrecision / (
                                                                            methodRecall + methodPrecision)

        methodMetrics = {'precision': methodPrecision,
                         'recall': methodRecall, 'hmean': methodHmean}

        return methodMetrics
        

def build_list_dict( cord_list ) :
  list = []
  for i in cord_list :
    list.append( {
          'points': [(i[0], i[1]), (i[2], i[3]), (i[4], i[5]), (i[6], i[7])],
          'text': '',
          'ignore': False,} )
  return list
  
  
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
    fontpath = 'simsun.ttc'
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
  
def init_args():
    parser = argparse.ArgumentParser(description='Map.DBNet')

    # input/output path
    parser.add_argument('--pred_folder', default='./test/input', type=str)
    parser.add_argument('--gt_folder', default='./test/output', type=str)
    parser.add_argument('--img_folder', default='./test/img', type=str)
    
    # visualize
    parser.add_argument('--result_visualize', action='store_true')
    parser.add_argument('--compare_group', action='store_true')
        
    args = parser.parse_args()
    return args
    
    
def read_label_xml_file( file_name ) :
  f = open( file_name + '.xml', 'r')

  for i in range(9) : # read garbage
    f.readline()

  filename = f.readline().split("'")[1]
  
  file_list = []

  lines = f.readlines()

  for i in range( 0, len(lines) - 4, 3 ) :

    temp = lines[i].split("'")
    if "B-3631-0062-137b_" not in file_name :
      file_list.append( [ int(temp[3]), int(temp[1]), int(temp[3])+int(temp[5]), int(temp[1]), int(temp[3])+int(temp[5]), int(temp[1])+int(temp[7]), int(temp[3]), int(temp[1])+int(temp[7]), (lines[i+1].split(">")[1]).split("<")[0], int(temp[8].split("\"")[1]) ] )
    else :
      file_list.append( [ int(temp[3]), int(temp[1]), int(temp[3])+int(temp[5]), int(temp[1]), int(temp[3])+int(temp[5]), int(temp[1])+int(temp[7]), int(temp[3]), int(temp[1])+int(temp[7]), ((lines[i+1].split(">")[1]).split("<")[0]).split(",")[0], int(((lines[i+1].split(">")[1]).split("<")[0]).split(",")[1]) ] )
    
  return file_list
  
if __name__ == '__main__':
    args = init_args()
    print(args)
    
    #read OCR output folder all txt file
    pred_folder = args.pred_folder
    pred_name_list = [ f for f in os.listdir( pred_folder ) if os.path.isfile(os.path.join( pred_folder, f )) and f.endswith( '.txt' ) ]
    print( pred_name_list )
  
    evaluator = DetectionIoUEvaluator( is_output_polygon = False )
    
    det_T_rec_T_list = []
    det_T_rec_F_list = []
    det_F_list = []
    det_M_list = []
    precision_list = []
    recall_list = []
    group_correct_list = []
    perfect_list = []
    d_hmean_list = []
    d_precision_list = []
    d_recall_list = []
    
    for pred_name_num in tqdm(range( len(pred_name_list) ) ) :
    
        det_T_rec_T = 0
        det_T_rec_F = 0
        det_F = 0
        det_M = 0
        
        # [(x,y)*4, word, group_id ]
        f = open( args.pred_folder + "/" + pred_name_list[pred_name_num], 'r',  encoding='utf16' )
        pred_cord = []
        for line in f.readlines():
            temp = line.split(",")[:8]
            temp = [ int(i) for i in temp ]
            temp.append( line.split(",")[-2] )
            temp.append( int(line.split(",")[-1][:-1]) )
            pred_cord.append(temp)

        tg_cord = read_label_xml_file( args.gt_folder + "/" + pred_name_list[pred_name_num][:-4] )
  
        result = evaluator.evaluate_image( build_list_dict(tg_cord), build_list_dict(pred_cord) )
        d_hmean_list.append(result["hmean"])
        d_precision_list.append(result["precision"])
        d_recall_list.append(result["recall"])
        
        if args.result_visualize :
        
            image = cv2.imread( args.img_folder + "/" + pred_name_list[pred_name_num][:-4] + ".jpg")
            
            for i in result["pairs"] :
                if pred_cord[i['det']][8] == tg_cord[i['gt']][8] :
                  det_T_rec_T +=1
                  image = drawbox( image, pred_cord[i['det']][0], pred_cord[i['det']][1], pred_cord[i['det']][2], pred_cord[i['det']][3], pred_cord[i['det']][4], pred_cord[i['det']][5], pred_cord[i['det']][6], pred_cord[i['det']][7], (0,0,255), 1, True, tg_cord[i['gt']][8] )
                else :
                  det_T_rec_F +=1
                  image = drawbox( image, pred_cord[i['det']][0], pred_cord[i['det']][1], pred_cord[i['det']][2], pred_cord[i['det']][3], pred_cord[i['det']][4], pred_cord[i['det']][5], pred_cord[i['det']][6], pred_cord[i['det']][7], (255,0,255), 1, True, pred_cord[i['det']][8] + "/" +tg_cord[i['gt']][8] )
      
            match_tg_index = [i['gt'] for i in result["pairs"] ]
            match_pred_index = [i['det'] for i in result["pairs"] ]
            match_tg_index.sort()
            for i in range( len( tg_cord ) ) :
              if i not in match_tg_index :
                det_M +=1
                image = drawbox( image, tg_cord[i][0], tg_cord[i][1], tg_cord[i][2], tg_cord[i][3], tg_cord[i][4], tg_cord[i][5], tg_cord[i][6], tg_cord[i][7], (0,255,0), 1, False, "" )
        
            for i in range( len( pred_cord ) ) :
              if i not in match_pred_index :
                det_F +=1
                image = drawbox( image, pred_cord[i][0], pred_cord[i][1], pred_cord[i][2], pred_cord[i][3], pred_cord[i][4], pred_cord[i][5], pred_cord[i][6], pred_cord[i][7], (255,0,0), 2, True, pred_cord[i][8] )
        
            cv2.imwrite( args.pred_folder + "/" + pred_name_list[pred_name_num][:-4] + "_eval_result.jpg", image )
            
        else :
            for i in result["pairs"] :
                if pred_cord[i['det']][8] == tg_cord[i['gt']][8] :
                  det_T_rec_T +=1
                else :
                  det_T_rec_F +=1
      
            match_tg_index = [i['gt'] for i in result["pairs"] ]
            match_pred_index = [i['det'] for i in result["pairs"] ]
            match_tg_index.sort()
            for i in range( len( tg_cord ) ) :
              if i not in match_tg_index :
                det_M +=1
        
            for i in range( len( pred_cord ) ) :
              if i not in match_pred_index :
                det_F +=1
        
        det_T_rec_T_list.append(det_T_rec_T)
        det_T_rec_F_list.append(det_T_rec_F)
        det_F_list.append(det_F)
        det_M_list.append(det_M)
        precision_list.append(det_T_rec_T/len( tg_cord ))
        recall_list.append(det_T_rec_T/len( pred_cord ))
        
        if args.compare_group :
        
            group_correct_id_list = []
            group_correct = 0
            group_correct_perfect = 0
            pred_group = [ [] for i in range( len( list( set( [i[9] for i in pred_cord ] ) ) ) ) ]
            tg_group = [ [] for i in range( len( list( set( [i[9] for i in tg_cord ] ) ) ) ) ]
            
            tg_set = list( set( [i[9] for i in tg_cord ] ) )
            for i in range(len(tg_cord)) :
                tg_cord[i][9] = tg_set.index( tg_cord[i][9] )
                tg_group[tg_cord[i][9]].append(i)

            for i in result["pairs"] :
                pred_group[pred_cord[i['det']][9]].append( (i['det'],i['gt']) )
            # print(tg_group)
            
            
            for i in range( len(pred_group) ) :
                if len( pred_group[i] ) != 0 and len(pred_group[i]) == len(tg_group[tg_cord[pred_group[i][0][1]][9]]) : #i非空,兩個長度是否一樣
                    is_group = True
                    for j in pred_group[i] :
                        if j[1] not in tg_group[tg_cord[pred_group[i][0][1]][9]] :
                            is_group = False
                            break
                    if is_group :
                        group_correct_id_list.append( i )
                        group_correct +=1
                    
            group_correct_list.append( group_correct / len(tg_group) )

            
            for i in group_correct_id_list :
                for j in pred_group[i] :
                    perfect_correct = True
                    if pred_cord[j[0]][8] != tg_cord[j[1]][8] :
                        perfect_correct = False
                if perfect_correct :
                         group_correct_perfect+=1
            
            perfect_list.append( group_correct_perfect / len(tg_group) )
              
              
    print( "det_T_rec_T : {}".format( np.mean(det_T_rec_T_list) ) )
    print( "det_T_rec_F : {}".format( np.mean(det_T_rec_F_list) ) )
    print( "det_F : {}".format( np.mean(det_F_list) ) )
    print( "det_M : {}".format( np.mean(det_M_list) ) )
    print( "precision : {}".format( np.mean(precision_list) ) )
    print( "recall : {}".format( np.mean(recall_list) ) )
    print( "f1 : {}".format( 2*np.mean(precision_list)*np.mean(recall_list) / (np.mean(precision_list)+np.mean(recall_list)) ) )
    f1_list = []
    for i in range( len( precision_list ) ) :
      f1_list.append( 2*precision_list[i]*recall_list[i] / (precision_list[i]+recall_list[i] ) )
    print( "f1 : {}".format( np.mean(f1_list) ) )
    print( "="*20 )
    print( "group correct (without recongnition) : {}".format( np.mean(group_correct_list) ) )
    print( "group correct (with recongnition) : {}".format( np.mean(perfect_list) ) )
    
    print( "precision : {}".format( np.mean(d_precision_list) ) )
    print( "recall : {}".format( np.mean(d_recall_list) ) )
    print( "f1 : {}".format( np.mean(d_hmean_list) ) )