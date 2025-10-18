import json
import os
import argparse
from PIL import Image
from tqdm import tqdm


def load_coco_json(json_path):
    """Cargar archivo JSON en formato COCO"""
    with open(json_path, 'r') as f:
        return json.load(f)


def save_coco_json(coco_data, output_path):
    """Guardar datos en formato COCO JSON"""
    with open(output_path, 'w') as f:
        json.dump(coco_data, f, indent=2)


def calculate_patches_positions(img_width, img_height, patch_width, patch_height, overlap):
    """
    Calcular las posiciones de los parches en la imagen
    Returns: lista de tuplas (x_min, y_min, x_max, y_max)
    """
    patches = []
    stride_x = patch_width - overlap
    stride_y = patch_height - overlap
    
    y = 0
    while y < img_height:
        x = 0
        while x < img_width:
            x_max = min(x + patch_width, img_width)
            y_max = min(y + patch_height, img_height)
            
            # Ajustar para que el parche tenga el tamaño correcto
            x_min = max(0, x_max - patch_width)
            y_min = max(0, y_max - patch_height)
            
            patches.append((x_min, y_min, x_max, y_max))
            
            x += stride_x
            if x >= img_width:
                break
        
        y += stride_y
        if y >= img_height:
            break
    
    return patches


def bbox_intersection(bbox1, bbox2):
    """
    Calcular intersección entre dos bounding boxes
    bbox format: [x_min, y_min, x_max, y_max]
    """
    x_min = max(bbox1[0], bbox2[0])
    y_min = max(bbox1[1], bbox2[1])
    x_max = min(bbox1[2], bbox2[2])
    y_max = min(bbox1[3], bbox2[3])
    
    if x_max <= x_min or y_max <= y_min:
        return None
    
    return [x_min, y_min, x_max, y_max]


def bbox_area(bbox):
    """Calcular área de un bounding box [x_min, y_min, x_max, y_max]"""
    return max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1])


def coco_to_xyxy(coco_bbox):
    """Convertir COCO bbox [x, y, w, h] a [x_min, y_min, x_max, y_max]"""
    x, y, w, h = coco_bbox
    return [x, y, x + w, y + h]


def xyxy_to_coco(xyxy_bbox):
    """Convertir [x_min, y_min, x_max, y_max] a COCO bbox [x, y, w, h]"""
    x_min, y_min, x_max, y_max = xyxy_bbox
    return [x_min, y_min, x_max - x_min, y_max - y_min]


def transform_bbox_to_patch(bbox_xyxy, patch_position):
    """
    Transformar bbox de coordenadas de imagen original a coordenadas del parche
    bbox_xyxy: [x_min, y_min, x_max, y_max] en imagen original
    patch_position: (patch_x_min, patch_y_min, patch_x_max, patch_y_max)
    """
    patch_x_min, patch_y_min, _, _ = patch_position
    
    new_bbox = [
        bbox_xyxy[0] - patch_x_min,
        bbox_xyxy[1] - patch_y_min,
        bbox_xyxy[2] - patch_x_min,
        bbox_xyxy[3] - patch_y_min
    ]
    
    return new_bbox


def create_patches_and_annotations(images_dir, json_file, output_dir, 
                                   patch_height, patch_width, overlap, 
                                   min_visibility=0.1):
    """
    Función principal para crear parches y ajustar anotaciones
    """
    # Crear directorio de salida
    os.makedirs(output_dir, exist_ok=True)
    
    # Cargar COCO JSON
    print("Cargando anotaciones COCO...")
    coco_data = load_coco_json(json_file)
    
    # Crear estructura para nuevo COCO JSON
    new_coco = {
        'info': coco_data.get('info', []),
        'licenses': coco_data.get('licenses', []),
        'images': [],
        'annotations': [],
        'categories': coco_data.get('categories', [])
    }
    
    new_coco['categories'].extend([{
        "id": 0,
        "name": "Background",
        "supercategory": "none"
        }])
    
    # Crear mapeos
    image_id_to_info = {img['id']: img for img in coco_data['images']}
    
    # Agrupar anotaciones por imagen
    annotations_by_image = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in annotations_by_image:
            annotations_by_image[img_id] = []
        annotations_by_image[img_id].append(ann)
    
    # Contadores para nuevos IDs
    new_image_id = 1
    new_annotation_id = 1
    
    # Procesar cada imagen
    print(f"\nProcesando {len(coco_data['images'])} imágenes...")
    for img_info in tqdm(coco_data['images']):
        img_id = img_info['id']
        img_filename = img_info['file_name']
        img_path = os.path.join(images_dir, img_filename)
        
        # Verificar que la imagen existe
        if not os.path.exists(img_path):
            print(f"Advertencia: Imagen no encontrada: {img_path}")
            continue
        
        # Abrir imagen
        img = Image.open(img_path)
        img_width, img_height = img.size
        
        # Obtener anotaciones de esta imagen
        img_annotations = annotations_by_image.get(img_id, [])
        
        # Calcular posiciones de parches
        patches_positions = calculate_patches_positions(
            img_width, img_height, patch_width, patch_height, overlap
        )
        
        # Procesar cada parche
        for patch_idx, patch_pos in enumerate(patches_positions):
            patch_x_min, patch_y_min, patch_x_max, patch_y_max = patch_pos
            
            # Crear nombre para el parche
            base_name, ext = os.path.splitext(img_filename)
            patch_filename = f"{base_name}_{patch_idx}{ext}"
            
            # Extraer parche de la imagen
            patch_img = img.crop((patch_x_min, patch_y_min, patch_x_max, patch_y_max))
            
            # Si el parche es más pequeño que el tamaño deseado, hacer padding
            if patch_img.size != (patch_width, patch_height):
                padded = Image.new('RGB', (patch_width, patch_height), (0, 0, 0))
                padded.paste(patch_img, (0, 0))
                patch_img = padded
            
            # Procesar anotaciones para este parche
            patch_annotations = []
            patch_bbox_xyxy = [patch_x_min, patch_y_min, patch_x_max, patch_y_max]
            
            for ann in img_annotations:
                # Convertir bbox de COCO a xyxy
                ann_bbox_xyxy = coco_to_xyxy(ann['bbox'])
                
                # Calcular intersección
                intersection = bbox_intersection(ann_bbox_xyxy, patch_bbox_xyxy)
                
                if intersection is None:
                    continue
                
                # Calcular visibilidad
                original_area = bbox_area(ann_bbox_xyxy)
                intersection_area = bbox_area(intersection)
                visibility = intersection_area / original_area if original_area > 0 else 0
                
                # Filtrar por visibilidad mínima
                if visibility < min_visibility:
                    continue
                
                # Transformar bbox a coordenadas del parche
                patch_bbox = transform_bbox_to_patch(intersection, patch_pos)
                
                # Convertir de vuelta a formato COCO
                patch_bbox_coco = xyxy_to_coco(patch_bbox)
                
                # Crear nueva anotación
                new_ann = {
                    'id': new_annotation_id,
                    'image_id': new_image_id,
                    'category_id': ann['category_id'],
                    'bbox': patch_bbox_coco,
                    'area': patch_bbox_coco[2] * patch_bbox_coco[3],
                    'iscrowd': ann.get('iscrowd', 0)
                }
                
                patch_annotations.append(new_ann)
                new_annotation_id += 1
            
            # Solo guardar parches que tienen anotaciones
            if len(patch_annotations) > 0:
                # Guardar imagen del parche
                patch_output_path = os.path.join(output_dir, patch_filename)
                patch_img.save(patch_output_path)
                
                # Agregar información de imagen al nuevo COCO
                new_coco['images'].append({
                    'id': new_image_id,
                    'file_name': patch_filename,
                    'width': patch_width,
                    'height': patch_height,
                    'base_image': img_filename,
                    'base_image_width': img_width,
                    'base_image_height': img_height,
                    'patch_position': patch_pos
                })
                
                # Agregar anotaciones
                new_coco['annotations'].extend(patch_annotations)
                
                new_image_id += 1
    
    # Guardar nuevo JSON
    output_json_path = os.path.join(output_dir, '_annotations.coco.json')
    save_coco_json(new_coco, output_json_path)
    
    print(f"\n Proceso completado!")
    print(f"   - Imágenes originales: {len(coco_data['images'])}")
    print(f"   - Parches creados: {len(new_coco['images'])}")
    print(f"   - Anotaciones originales: {len(coco_data['annotations'])}")
    print(f"   - Anotaciones en parches: {len(new_coco['annotations'])}")
    print(f"   - Directorio de salida: {output_dir}")
    print(f"   - JSON de salida: {output_json_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Crear parches de imágenes y ajustar anotaciones COCO'
    )
    
    parser.add_argument('--images_dir', type=str, required=True,
                        help='Directorio con las imágenes originales')
    parser.add_argument('--json_file', type=str, required=True,
                        help='Archivo JSON con anotaciones COCO')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directorio de salida para parches y JSON')
    parser.add_argument('--patch_height', type=int, default=256,
                        help='Altura de los parches en píxeles (default: 256)')
    parser.add_argument('--patch_width', type=int, default=256,
                        help='Ancho de los parches en píxeles (default: 256)')
    parser.add_argument('--overlap', type=int, default=50,
                        help='Solapamiento entre parches en píxeles (default: 50)')
    parser.add_argument('--min_visibility', type=float, default=0.1,
                        help='Fracción mínima de visibilidad para mantener anotación (default: 0.1)')
    
    args = parser.parse_args()
    
    # Validar argumentos
    if not os.path.exists(args.images_dir):
        print(f"Error: Directorio de imágenes no existe: {args.images_dir}")
        return
    
    if not os.path.exists(args.json_file):
        print(f"Error: Archivo JSON no existe: {args.json_file}")
        return
    
    if args.patch_height <= 0 or args.patch_width <= 0:
        print(f"Error: Dimensiones de parche deben ser positivas")
        return
    
    if args.overlap < 0:
        print(f"Error: Solapamiento no puede ser negativo")
        return
    
    if args.min_visibility < 0 or args.min_visibility > 1:
        print(f"Error: min_visibility debe estar entre 0 y 1")
        return
    
    # Ejecutar proceso
    create_patches_and_annotations(
        images_dir=args.images_dir,
        json_file=args.json_file,
        output_dir=args.output_dir,
        patch_height=args.patch_height,
        patch_width=args.patch_width,
        overlap=args.overlap,
        min_visibility=args.min_visibility
    )


if __name__ == '__main__':
    main()

