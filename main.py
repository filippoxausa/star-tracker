import cv2
import os
import glob
import numpy as np

def get_sorted_images(image_folder):
    search_path = os.path.join(image_folder, "*.jpg") 
    files = glob.glob(search_path)

    files.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    return files

def task_1_visualization(image_folder):
    files = get_sorted_images(image_folder)
    
    if not files:
        print("Nessuna immagine trovata.")
        return
    
    print(f"Visualizzazione ({len(files)} immagini)")

    for filename in files:
        img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        if img is None: continue
        
        cv2.imshow("Raw Sequence", img)
        
        cv2.waitKey(100)
            
    cv2.destroyAllWindows()

def task_2_denoising(image_folder):
    files = get_sorted_images(image_folder)
    
    if not files:
        print("Nessuna immagine trovata.")
        return

    print(f"Pulizia Rumore ({len(files)} immagini)")

    # kernel per morfologia (3x3 pixel)
    kernel = np.ones((3,3), np.uint8)

    for filename in files:
        img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        if img is None: continue

        # per lavorare con nero "puro"
        _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

        # Operazione apertura (rimuove i punti più piccoli del kernel)
        clean_img = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        final_img = cv2.morphologyEx(clean_img, cv2.MORPH_CLOSE, kernel, iterations=1)

        combined = np.hstack((img, final_img))
        
        cv2.putText(combined, "Originale", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.putText(combined, "Pulita", (img.shape[1] + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        
        cv2.imshow("Denoising Comparison", combined)
        
        if cv2.waitKey(0) == ord('q'):
            break
            
    cv2.destroyAllWindows()



if __name__ == "__main__":
    folder = "images"

    task_1_visualization(folder)
    task_2_denoising(folder)
