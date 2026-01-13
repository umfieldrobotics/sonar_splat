import torch    
import numpy as np 
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
import os 
import cv2
import pandas as pd
from lpipsPyTorch import lpips
import tqdm
import argparse
from collections import defaultdict
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#write a function that reads in all the images in a gt folder and int eh prediction folder and calculates these metrics and averages them 
def evaluate_imgs(gt_folder, pred_folder):
    #read in all the images in the gt folder
    #read in all the images in the pred folder
    pred_images = []
    psnrs_vals = []
    ssims_vals = []
    lpips_vals = []
    # lpips = LearnedPerceptualImagePatchSimilarity(
    #             net_type="vgg", normalize=True
    #         ).to(device)
    if len(os.listdir(pred_folder)) != len(os.listdir(gt_folder)):
        print(f"\033[91mPred folder {pred_folder.split('/')[-5]} size != GT folder {gt_folder.split('/')[-5]} size\033[0m")
        return 0, 0, 0
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)

    for file in os.listdir(pred_folder):
        pred_image = cv2.imread(os.path.join(pred_folder, file))
        gt_image = cv2.imread(os.path.join(gt_folder, file))

        pred_image = pred_image.transpose(2, 0, 1)/255
        gt_image = gt_image.transpose(2, 0, 1)/255

        pred_image = remove_border_pixels(pred_image)
        gt_image = remove_border_pixels(gt_image)
   
        #calculate the metrics
        lpips_vals.append(lpips(torch.from_numpy(pred_image).unsqueeze(0).float().to(device), torch.from_numpy(gt_image).unsqueeze(0).float().to(device), net_type="vgg").item())
        ssims_vals.append(ssim(torch.from_numpy(pred_image).unsqueeze(0).float().to(device), torch.from_numpy(gt_image).unsqueeze(0).float().to(device)).item())
        psnrs_vals.append(psnr(torch.from_numpy(pred_image).unsqueeze(0).float().to(device), torch.from_numpy(gt_image).unsqueeze(0).float().to(device)).item())
        # lpips_vals.append(0)
    return np.mean(lpips_vals), np.mean(ssims_vals), np.mean(psnrs_vals)

def verify_and_evaluate(root_folder, validate_only=False): 
    methods = os.listdir(root_folder)
    all_results = {}

    method_to_gt_folder_list = defaultdict(dict)
    for method in methods:
        print(f"Evaluating {method}")
        scenes = os.listdir(os.path.join(root_folder, method))
        #sort scenes by number
        scenes.sort()
        #store values for each scene in a dictionary  and write to .csv file with scene name as the first column
        results = {}
        results["Metric"] = ["PSNR", "SSIM", "LPIPS"]
        results["method"] = method
        gt_folder_dict = {}
        for scene in tqdm.tqdm(scenes):
            if "pole_qual1" in scene:
                if method.lower() == "zsplat":
                    gt_folder = os.path.join(root_folder, method, scene, "test/ours_30000/gt")
                    pred_folder = os.path.join(root_folder, method, scene, "test/ours_30000/renders")
                else:
                    gt_folder = os.path.join(root_folder, method, scene, "sonar_renders/test/gt_sonar_images")
                    pred_folder = os.path.join(root_folder, method, scene, "sonar_renders/test/sonar_images")
                num_imgs = len(os.listdir(gt_folder))
                gt_folder_dict[scene] = gt_folder
                if not validate_only:
                    lpips_average, ssim_average, psnr_average = evaluate_imgs(gt_folder, pred_folder)
                else:
                    lpips_average, ssim_average, psnr_average = 0, 0, 0
                print(f"LPIPS: {lpips_average}, SSIM: {ssim_average}, PSNR: {psnr_average}")
                results[scene] = [psnr_average, ssim_average, lpips_average]
        
        #write all methods to one single csv file 
        results[f"Average"] = [np.mean([results[scene][0] for scene in results if scene != 'method' and scene != 'Metric']), 
                              np.mean([results[scene][1] for scene in results if scene != 'method' and scene != 'Metric']), 
                              np.mean([results[scene][2] for scene in results if scene != 'method' and scene != 'Metric'])]
        
        method_to_gt_folder_list[method] = gt_folder_dict
        df = pd.DataFrame(results)  
        all_results[method] = df
    #write all results to a single csv file 
    # for method_name, df in all_results.items():
    #     df["method"] = method_name
    combined_df = pd.concat(all_results.values(), ignore_index=True)
    combined_df.to_csv("all_results.csv", index=False)

    for method1 in method_to_gt_folder_list:
        for method2 in method_to_gt_folder_list:
            intersection_scenes = list(set(method_to_gt_folder_list[method1]) & set(method_to_gt_folder_list[method2]))
            for scene in intersection_scenes:
                validate_images_two_folders(method_to_gt_folder_list[method1][scene], method_to_gt_folder_list[method2][scene])

def remove_border_pixels(img):
    img = img[:, 10:-10, 10:-10]
    return img

def validate_images_two_folders(folder1, folder2):
    #read all images and verify that the images are the same  
    img_paths1 = os.listdir(folder1)
    img_paths2 = os.listdir(folder2)
    if len(img_paths1) != len(img_paths2):
        #make red 
        print(f"\033[91mThe folder {folder1.split('/')[-5]} does not have the same number of images as {folder2.split('/')[-5]} -> {folder1.split('/')[-4]}\033[0m")
        return False
    
    #sort the paths regardless of leading zeros 
    img_paths1.sort(key=lambda x: int(x.split('.')[0]))
    img_paths2.sort(key=lambda x: int(x.split('.')[0]))

    #go through all the folders and verify that the number of images are the same 
    error = False
    for img_idx in range(len(img_paths1)):
        img1_fullpath = os.path.join(folder1, img_paths1[img_idx])
        img2_fullpath = os.path.join(folder2, img_paths2[img_idx])
        # if img_paths1[img_idx] != img_paths2[img_idx]:
        #     print(f"The image {img1_fullpath} does not have the same name as {img2_fullpath}")
        #     error = True
        img1 = cv2.imread(img1_fullpath)
        img2 = cv2.imread(img2_fullpath)

        img1 = remove_border_pixels(img1)
        img2 = remove_border_pixels(img2)



        if img1.shape != img2.shape:
            #print in red 
            print(f"\033[91mThe image {img1_fullpath} does not have the same shape as {img2_fullpath}\033[0m")
            error = True
        if np.array_equal(img1, img2) == False: 
            #print in red 
            print(f"\033[91mThe image from {img1_fullpath.split('/')[-6]} \
                  does not match {img2_fullpath.split('/')[-6]} -> {img1_fullpath.split('/')[-5]}, img: {img_paths1[img_idx]}\033[0m")
            error = True
    
    scene1 = img1_fullpath.split('/')[-5]
    scene2 = img2_fullpath.split('/')[-5]
    if scene1 != scene2:
        print(f"\033[91mOops, naming convention off for datasets {img1_fullpath.split('/')[-6]} and {img2_fullpath.split('/')[-6]}!\033[0m")
        error = True
    if not error: 
        #print this in green 
        print(f"\033[92m{img1_fullpath.split('/')[-6]} and {img2_fullpath.split('/')[-6]} are the same -> {scene1}!\033[0m")

    return not error

def main():
    #parse args 
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_folder", type=str, default="./")
    parser.add_argument("--validate_only", type=bool, default=False)
    
    args = parser.parse_args()
    verify_and_evaluate(args.root_folder, args.validate_only)

if __name__ == "__main__":
    main()

