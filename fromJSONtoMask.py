def load_gen_v4(image_path, gt_path, img_files):#, batch_size, n_classes):
	image_files = os.listdir(image_path)

	# carico file via - load annotations
	annotations = json.load(open(os.path.join(gt_path, "via_region_data.json")))
	annotations = list(annotations.values())  # don't need the dict keys
    # The VIA tool saves images in the JSON even if they don't have any
    # annotations. Skip unannotated images.
	annotations = [a for a in annotations if a['regions'] ]

	# carico immagini
	for a in annotations:
        # Get the x, y coordinaets of points of the polygons that make up
        # the outline of each object instance. These are stores in the
        # shape_attributes (see json format above)
        # The if condition is needed to support VIA versions 1.x and 2.x.
		if type(a['regions']) is dict:
			polygons = [r['shape_attributes'] for r in a['regions'].values()]
		else:
			polygons = [r['shape_attributes'] for r in a['regions']]

        # load_mask() needs the image size to convert polygons to masks.
        # Unfortunately, VIA doesn't include it in JSON, so we must read
        # the image. This is only managable since the dataset is tiny.
		image_path2 = os.path.join(image_path, a['filename'])
		image = skimage.io.imread(image_path2)
		height, width = image.shape[:2]

		# genero le maschere from annotations e vettore con le classi, ora solo 1
		img_masks, class_ids = load_mask(width, height, polygons, a['filename'])

		yield get_batch(image, img_masks)

def load_mask(width, height, polygons, image_id):
	"""Generate instance masks for an image.
   Returns:
	masks: A bool array of shape [height, width] with
		one mask per instance.
	class_ids: a 1D array of class IDs of the instance masks.
	"""
	# Convert polygons to a bitmap mask of shape
	# [height, width]
	# 1 for mask, 0 for other
	mask = np.zeros([height, width], dtype=np.uint8)

	for i, p in enumerate(polygons):
		# Get indexes of pixels inside the polygon and set them to 1
		rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
		mask[rr, cc] = 1

	# Return mask, and array of class IDs. Since we have
	# one class ID only, we return an array of 1s
	return mask, np.ones([mask.shape[-1]], dtype=np.int32)
