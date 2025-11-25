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
        
        if cv2.waitKey(0) == ord('q'):
            break
            
    cv2.destroyAllWindows()


def task_2_denoising(image_folder):
    files = get_sorted_images(image_folder)
    
    if not files:
        print("Nessuna immagine trovata.")
        return []

    print(f"Pulizia Rumore ({len(files)} immagini)")

    # kernel per morfologia (3x3 pixel)
    kernel = np.ones((3,3), np.uint8)
    
    cleaned_images = []

    for filename in files:
        img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        if img is None: continue

        # per lavorare con nero "puro", da scala di grigi a binario
        _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

        # operazione apertura (effettua erosione)
        clean_img = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
		# operazione chiusura (effettua dilatazione)
        final_img = cv2.morphologyEx(clean_img, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        cleaned_images.append(clean_img)

        combined = np.hstack((img, clean_img))
        
        cv2.putText(combined, "Originale", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.putText(combined, "Pulita", (img.shape[1] + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        
        cv2.imshow("Denoising Comparison", combined)
        
        if cv2.waitKey(0) == ord('q'):
            break
            
    cv2.destroyAllWindows()

    return cleaned_images


def task_3_star_detection(cleaned_images):
    if not cleaned_images: 
        print("Nessuna immagine pulita da processare.")
        return

    
    output_filename = "bounding_boxes.txt"
    print(f"Rilevamento Bounding Box e Centroidi ({len(cleaned_images)} immagini)")

    with open(output_filename, 'w') as f:
        for idx, clean_img in enumerate(cleaned_images):
            
            # cv2.RETR_EXTERNAL: mi interessano solo i contorni esterni (non eventuali buchi all'interno)
            # cv2.CHAIN_APPROX_SIMPLE: comprime i vertifici significativi
            contours, _ = cv2.findContours(clean_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # creo una copia a colori (BGR) per poter disegnare i rettangoli della bounding box
            output_img = cv2.cvtColor(clean_img, cv2.COLOR_GRAY2BGR)

            # lista per salvare le coordinate di questo frame: [x1, y1, x2, y2] per ogni stella
            frame_data = []

            print(f"Immagine {idx} - Stelle trovate: {len(contours)}")

            for contour in contours:
                # x, y angoli in alto a sinistra
                # w, h larghezza e altezza
                x, y, w, h = cv2.boundingRect(contour)

                x2 = x + w
                y2 = y + h

                frame_data.append([x, y, x2, y2])
                
                # disegna un rettangolo verde intorno al contorno trovato
                
                
                M = cv2.moments(contour)
                # controllo M.m00 = area non nulla
                if M["m00"] >= 9:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    cv2.rectangle(output_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.circle(output_img, (cX, cY), 2, (0, 0, 255), -1)
                
            
            f.write(" ".join(str(bbox) for bbox in frame_data) + "\n")
            cv2.imshow("Star Detection", output_img)
            
            if cv2.waitKey(0) == ord('q'):
                break
            
    cv2.destroyAllWindows()



if __name__ == "__main__":
    folder = "images"

    task_1_visualization(folder)
    cleaned_images = task_2_denoising(folder)
    task_3_star_detection(cleaned_images)
