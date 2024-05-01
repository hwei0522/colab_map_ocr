import os
import cv2
import torch
import argparse
import pathlib
import numpy as np
import shutil

from PIL import Image
from tqdm import tqdm
from datetime import datetime
from utils.generate_group_file import generate_file
from utils.util_myself import crop_image, merge_txt_file, xml_output, drawbox, remove_overlap, remove_num_char

# detection package
from detection_model.pytorch_build_model import Pytorch_model
from utils.util import draw_bbox, save_result, get_file_list

# recognition package
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from dataset import decode_text

# place name grouping
from group_place import group_place_name

def init_args():
    parser = argparse.ArgumentParser(description='Map.DBNet')

    # input/output path
    parser.add_argument('--input_folder', default='./test/input', type=str)
    parser.add_argument('--output_folder', default='./test/output', type=str)

    # crop image size
    parser.add_argument('--output_width', default=1500, type=int)
    parser.add_argument('--output_height', default=1500, type=int)
    parser.add_argument('--overlap_length', default=100, type=int)

    # model path
    parser.add_argument('--detection_model_path', default=r'model_best.pth', type=str)
    parser.add_argument('--recognition_model_path', default=r'model_best.pth', type=str)

    # detection
    parser.add_argument('--thre', default=0.3, type=float, help='the thresh of post_processing')
    parser.add_argument('--polygon', action='store_true', help='output polygon or box')
    parser.add_argument('--save_result', action='store_true')

    # recognition
    parser.add_argument('--CUDA_VISIBLE_DEVICES', default='-1', type=str)
    
    # coordinate convert
    parser.add_argument('--coordinate_convert', action='store_true')
    # parser.add_argument('--jgw_path', default='./test/jgw_path', type=str)
    
    # visualize
    parser.add_argument('--result_visualize', action='store_true')
    
    args = parser.parse_args()
    return args

        
if __name__ == '__main__':
    args = init_args()
    print(args)
    
    os.makedirs( args.output_folder, exist_ok=True )
    os.makedirs( args.output_folder + "/detection", exist_ok=True )
    os.makedirs( args.output_folder + "/ocr", exist_ok=True )

    #read input folder all image name
    img_path = args.input_folder
    img_name_list = [ f for f in os.listdir( img_path ) if os.path.isfile(os.path.join( img_path, f )) and f.endswith( '.jpg' ) ]

    # detection model setup
    detect_model = Pytorch_model( args.detection_model_path, post_p_thre=args.thre, gpu_id = 0 )

    # recognition model setup
    processor = TrOCRProcessor.from_pretrained(args.recognition_model_path)
    vocab = processor.tokenizer.get_vocab()
    vocab_inp = {vocab[key]: key for key in vocab}
    recognition_model = VisionEncoderDecoderModel.from_pretrained(args.recognition_model_path)
    recognition_model.eval()
    vocab = processor.tokenizer.get_vocab()
    vocab_inp = {vocab[key]: key for key in vocab}


    for i in range( len(img_name_list) ) :
        start = datetime.now()
        print( "\nPredicting image : " + img_name_list[i] )
        # 1. crop small image
        # input : full image
        # output : save crop image in temp folder
        os.makedirs( args.input_folder + "/temp", exist_ok=True )
        crop_image( args.input_folder, img_name_list[i], args.input_folder + "/temp", args.output_width, args.output_height, args.overlap_length )

        # 2. detection
        # input : temp folder small image
        # output : save detection txt in "{output_folder}/detection", file name end with "{image_name}_{id}.txt"
        print( "* Text detecting ~" )
        for img_path in tqdm( get_file_list(args.input_folder + "/temp", p_postfix=['.jpg'] ) ):
            preds, boxes_list, score_list, t = detect_model.predict( img_path, is_output_polygon=args.polygon )
            img = draw_bbox( cv2.imread( img_path )[:, :, ::-1], boxes_list )

            img_path = pathlib.Path( img_path )
            output_path = os.path.join( args.output_folder + "/detection", img_path.stem + '_result.jpg' )
            # pred_path = os.path.join( args.output_folder + "/detection", img_path.stem + '_pred.jpg' )
            # cv2.imwrite( output_path, img[:, :, ::-1] )
            # cv2.imwrite( pred_path, preds * 255 )
            save_result( output_path.replace('_result.jpg', '.txt'), boxes_list, score_list, args.polygon )

        shutil.rmtree( args.input_folder + "/temp" )

        # 3. merge detection txt file
        # input : merge in "{output_folder}/detection" small result
        # output : save merge full image detection txt in "{output_folder}/detection", file name as "{image_name}.txt"
        merge_label = merge_txt_file( args.output_folder + "/detection", img_name_list[i], args.output_folder + "/detection", args.output_width, args.overlap_length, args.input_folder )

        # 4. recognition
        # input : "{image_name}.txt" in "{output_folder}/detection"
        # output : save recognition txt for each line in recognition_list
        print( "* Text recognizing ~" )
        recognition_list = []

        img = Image.open( args.input_folder + "/" + img_name_list[i] ).convert('RGB')
        
        #num = 0
        x_max_list = []
        x_min_list = []
        y_max_list = []
        y_min_list = []
        for j in tqdm( merge_label ) :
            x_max = max(int(j[0]), int(j[2]), int(j[4]), int(j[6]))
            x_min = min(int(j[0]), int(j[2]), int(j[4]), int(j[6]))
            y_max = max(int(j[1]), int(j[3]), int(j[5]), int(j[7]))
            y_min = min(int(j[1]), int(j[3]), int(j[5]), int(j[7]))
            x_max_list.append( x_max )
            x_min_list.append( x_min )
            y_max_list.append( y_max )
            y_min_list.append( y_min )
            #img.crop((x_min, y_min, x_max, y_max)).save( args.output_folder + "/ocr/" + str(num) + ".jpg" )
            pixel_values = processor([img.crop((x_min, y_min, x_max, y_max))], return_tensors="pt").pixel_values
            with torch.no_grad():
                generated_ids = recognition_model.generate(pixel_values[:, :, :].cpu())

            recognition_list.append( decode_text(generated_ids[0].cpu().numpy(), vocab, vocab_inp) )
            #num+=1

        # 5. post process (with reshape to rectangle, remove latest two percentage boxes, remove overlap)
        # input : "{image_name}.txt" in "{output_folder}/detection" and recognition_list
        # output : post_process box_list, recognition_list
        area_list = [(x_max_list[j]-x_min_list[j])*(y_max_list[j]-y_min_list[j]) for j in range( len( merge_label ) )]
        thres = np.percentile( area_list, 4 )
        # std_v = np.std( area_list )
        for j in range( len(area_list) ) : 
          if area_list[j] <= thres :
            area_list[j] = True
          else :
            area_list[j] = False

        for j in  range( len(area_list)-1, -1, -1 ) :
          if area_list[j] :
            del x_max_list[j]
            del x_min_list[j]
            del y_max_list[j]
            del y_min_list[j]
            del recognition_list[j]
            del merge_label[j]
        
        x_max_list, x_min_list, y_max_list, y_min_list, recognition_list, merge_label = remove_overlap( x_max_list, x_min_list, y_max_list, y_min_list, recognition_list, merge_label )
        x_max_list, x_min_list, y_max_list, y_min_list, recognition_list, merge_label = remove_num_char( x_max_list, x_min_list, y_max_list, y_min_list, recognition_list, merge_label )
        
        # 6. group place name
        # input : "{image_name}.txt" in "{output_folder}/detection" and recognition_list
        # output : group_list
        print( "* Text grouping ~" )
        group, group_num = group_place_name( x_min_list, y_min_list, x_max_list, y_max_list )

        shutil.rmtree( args.output_folder + "/detection" )
        
        # 7. standard output
        # input : "{image_name}.txt" in "{output_folder}/detection" and recognition_list
        # output : save standard output txt in "{output_folder}/ocr"
        f = open( args.output_folder + "/ocr/" + img_name_list[i][:-4] + ".txt", "w", encoding='utf16' )
        for j in range( len( merge_label ) ) :
            f.write( str(x_min_list[j]) + "," + str(y_min_list[j]) + "," + \
                     str(x_max_list[j]) + "," + str(y_min_list[j]) + "," + \
                     str(x_max_list[j]) + "," + str(y_max_list[j]) + "," + \
                     str(x_min_list[j]) + "," + str(y_max_list[j]) + "," + \
                     str(merge_label[j][8][:-2]) + "," + recognition_list[j] + "," + \
                     str(group[j]) + "\n" )
        f.close()

        # 8. pretty xml output
        d_score = [j[8][:-2] for j in merge_label ]
        xml_output( args.output_folder + "/ocr/", args.coordinate_convert, args.input_folder, img_name_list[i], x_min_list, y_min_list, x_max_list, y_max_list, d_score, group, recognition_list, 0 )
        
        # 9. generate group file
        generate_file( img_name_list[i], args.input_folder, args.output_folder, args.result_visualize )

        # 10. result visualize
        if args.result_visualize :
            image = cv2.imread( args.input_folder + "/" + img_name_list[i] )
            for j in range( len( x_min_list ) ) :
                image = drawbox( image, x_min_list[j], y_min_list[j], x_max_list[j], y_min_list[j], x_max_list[j], y_max_list[j], x_min_list[j], y_max_list[j], (0,255,0), 1, True, str(recognition_list[j]) )
            cv2.imwrite( args.output_folder + "/ocr/" + img_name_list[i][:-4] + "_ocr.jpg", image )
        
        end = datetime.now()
        print('- Total number of text detections : ' + str( len( x_min_list ) ) )
        print('- Total number of group : ' + str( group_num ) )
        print('- Execution time : ' + str( datetime.strptime(str(end - start)[:-7], "%H:%M:%S") )[14:] )