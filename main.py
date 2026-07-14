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
        
        if cv2.waitKey(50) == ord('q'):
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

        # operazione apertura (effettua erosione seguita da dilatazione)
        clean_img = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
		# operazione chiusura (effettua dilatazione seguita da erosione)
        # final_img = cv2.morphologyEx(clean_img, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        cleaned_images.append(clean_img)

        combined = np.hstack((img, clean_img))
        
        cv2.putText(combined, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.putText(combined, "Filtered", (img.shape[1] + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        
        cv2.imshow("Denoising Comparison", combined)
        
        if cv2.waitKey(50) == ord('q'):
            break
            
    cv2.destroyAllWindows()

    return cleaned_images


def task_3_star_detection(cleaned_images):
    if not cleaned_images: 
        print("Nessuna immagine pulita da processare.")
        return []

    
    output_filename = "result/bounding_boxes.txt"
    print(f"Rilevamento Bounding Box e Centroidi ({len(cleaned_images)} immagini)")

    all_centroids = []

    with open(output_filename, 'w') as f:
        for idx, clean_img in enumerate(cleaned_images):
            
            # cv2.RETR_EXTERNAL: mi interessano solo i contorni esterni (non eventuali buchi all'interno)
            # cv2.CHAIN_APPROX_SIMPLE: comprime i vertifici significativi
            contours, _ = cv2.findContours(clean_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # creo una copia a colori (BGR) per poter disegnare i rettangoli della bounding box
            output_img = cv2.cvtColor(clean_img, cv2.COLOR_GRAY2BGR)

            # lista per salvare le coordinate di questo frame: [x1, y1, x2, y2] per ogni stella
            frame_data = []
            frame_centroids = []

            for contour in contours:
                # x, y angoli in alto a sinistra
                # w, h larghezza e altezza
                x, y, w, h = cv2.boundingRect(contour)

                x2 = x + w
                y2 = y + h
                
                M = cv2.moments(contour)
                # controllo M.m00 per evitare oggetti troppo piccoli (falsi positivi)
                if M["m00"] >= 9:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    cv2.rectangle(output_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.circle(output_img, (cX, cY), 2, (0, 0, 255), -1)
                    frame_data.append([x, y, x2, y2])
                    frame_centroids.append((cX, cY))
                
            print(f"Immagine {idx} - Stelle trovate: {len(frame_data)}")

            all_centroids.append(frame_centroids)
            
            f.write(" ".join(str(bbox) for bbox in frame_data) + "\n")
            cv2.imshow("Star Detection", output_img)
            
            if cv2.waitKey(50) == ord('q'):
                break
            
    cv2.destroyAllWindows()
    return all_centroids

def task_4_motion_tracking(all_centroids):
    if not all_centroids or len(all_centroids) < 2:
        print("Non ci sono abbastanza frame per calcolare il movimento.")
        return

    output_filename = "result/motion_log.txt"
    print(f"Calcolo Odometria (Nearest Neighbor + Mediana)")
    
    with open(output_filename, "w") as f:
        for i in range(1, len(all_centroids)):
            prev_frame = all_centroids[i-1]
            curr_frame = all_centroids[i]
            
            if not prev_frame or not curr_frame:
                f.write("0 0\n")
                print(f"Frame {i}: Nessuna stella rilevata, salto.")
                continue

            deltas_x = []
            deltas_y = []

            for (cx, cy) in curr_frame:
                min_dist = float('inf')
                best_match = None
                
                for (px, py) in prev_frame:
                    dist = np.sqrt((cx - px)**2 + (cy - py)**2)
                    
                    if dist < min_dist:
                        min_dist = dist
                        best_match = (px, py)
                
                if best_match:
                    dx = cx - best_match[0]
                    dy = cy - best_match[1]
                    deltas_x.append(dx)
                    deltas_y.append(dy)

            # utilizzo la MEDIANA (np.median) invece della media
            # per eliminare gli outlier
            if deltas_x:
                final_dx = np.median(deltas_x)
                final_dy = np.median(deltas_y)
            else:
                final_dx, final_dy = 0, 0

            f.write(f"x:{int(final_dx):>3d}; y:{int(final_dy):>3d};\t dist:({np.sqrt(final_dx**2 + final_dy**2):.2f})\n")
            
            # print(f"Frame {i}: Spostamento rilevato ({final_dx:.1f}, {final_dy:.1f})")

    print(f"Motion tracking completato: {output_filename} generato.")


if __name__ == "__main__":
    folder = "images"
    results_folder = "result"
    os.makedirs(results_folder, exist_ok=True)

    task_1_visualization(folder)
    cleaned_images = task_2_denoising(folder)
    all_centroids = task_3_star_detection(cleaned_images)
    task_4_motion_tracking(all_centroids)
