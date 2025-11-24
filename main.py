import cv2
import os
import glob
import numpy as np

def view_sequence(image_folder):
    search_path = os.path.join(image_folder, "*.jpg") 
    files = glob.glob(search_path)
    
    # ordina per numero invece che alfabeticamente
	# altrimenti: 0.jpg, 1.jpg, 10.jpg, 11.jpg, 2.jpg, 3.jpg
    files.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0])) 
    
    if not files:
        print("Errore: nessuna immagine trovata")
        return

    print(f"Trovate {len(files)} immagini")

    for filename in files:
		# grayscale in quanto sono immagini binarie (non a colori)
        img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            print(f"Errore nel caricamento di: {filename}")
            continue

        cv2.imshow("Star Tracker", img)
        
        key = cv2.waitKey(100) 
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    view_sequence("images")

	# tentativo con filtro canny → ottengo i bordi (scomodo lavorarci)
	# bocciato, provo altro metodo
