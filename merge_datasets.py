import os
import shutil
import yaml


# ============================================================
# SMART WASTE DATASET MERGER
# ============================================================

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_ROOT = os.path.join(
    PROJECT_DIR,
    "Dataset"
)

OUTPUT_ROOT = os.path.join(
    DATASET_ROOT,
    "Final_merged_dataset",
    "merged"
)


# ============================================================
# FINAL CLASSES
# ============================================================

TARGET_CLASSES = [
    "E_waste",
    "Medical_waste",
    "Hazardous_waste",
    "Chemical_waste",
    "Plastic_waste",
    "Paper_waste"
]


TARGET_IDS = {
    "E_waste": 0,
    "Medical_waste": 1,
    "Hazardous_waste": 2,
    "Chemical_waste": 3,
    "Plastic_waste": 4,
    "Paper_waste": 5
}


# ============================================================
# OLD DATASET-SPECIFIC MAPPING
# ============================================================

DATASET_CONFIG = {

    "Balanced E-Waste Dataset.yolov11": {
        "default": "E_waste",
        "mapping": {
            "Battery": "Hazardous_waste",
            "Straight-Tube-Fluorescent-Lamp": "Hazardous_waste"
        }
    },


    "GARBAGE CLASSIFICATION 3.yolov11": {
        "default": None,
        "mapping": {
            "PLASTIC": "Plastic_waste",
            "PAPER": "Paper_waste"
        }
    },


    "Plastic Waste.yolov11": {
        "default": None,
        "mapping": {
            "Plastic Bag": "Plastic_waste",
            "Plastic Bottle": "Plastic_waste",
            "Plastic Glass": "Plastic_waste"
        }
    },


    "bio medical waste classification.yolov11": {
        "default": None,
        "mapping": {
            "ampoule": "Medical_waste",
            "gloves": "Medical_waste",
            "mask": "Medical_waste",
            "metal": "Medical_waste"
        }
    },


    "chemical.yolov11": {
        "default": None,
        "mapping": {
            "spill": "Chemical_waste",
            "chemical": "Chemical_waste",
            "chemical waste": "Chemical_waste"
        }
    },


    "electronic-waste-dataset.yolov11": {
        "default": "E_waste",
        "mapping": {
            "thermometer": "Hazardous_waste"
        }
    },


    "medical waste.yolov11": {
        "default": None,
        "mapping": {
            "0": "Medical_waste",
            "syringe n glass": "Medical_waste"
        }
    }
}


# ============================================================
# AUTOMATIC MAPPING FOR NEW DATASETS
# ============================================================

def auto_map_class(source_name):

    name = normalize(source_name)

    # --------------------------------------------------------
    # PAPER
    # --------------------------------------------------------

    paper_keywords = [
        "paper",
        "newspaper",
        "cardboard",
        "carton",
        "magazine",
        "document",
        "notebook",
        "book"
    ]

    for keyword in paper_keywords:
        if keyword in name:
            return "Paper_waste"


    # --------------------------------------------------------
    # CHEMICAL
    # --------------------------------------------------------

    chemical_keywords = [
        "chemical",
        "chemical waste",
        "chemical_waste",
        "spill",
        "acid",
        "solvent",
        "laboratory chemical",
        "hazardous chemical"
    ]

    for keyword in chemical_keywords:
        if keyword in name:
            return "Chemical_waste"


    # --------------------------------------------------------
    # PLASTIC
    # --------------------------------------------------------

    plastic_keywords = [
        "plastic",
        "plastic bag",
        "plastic bottle",
        "plastic glass",
        "polythene"
    ]

    for keyword in plastic_keywords:
        if keyword in name:
            return "Plastic_waste"


    # --------------------------------------------------------
    # MEDICAL
    # --------------------------------------------------------

    medical_keywords = [
        "medical",
        "medical waste",
        "biomedical",
        "bio medical",
        "syringe",
        "glove",
        "mask",
        "ampoule"
    ]

    for keyword in medical_keywords:
        if keyword in name:
            return "Medical_waste"


    # --------------------------------------------------------
    # HAZARDOUS
    # --------------------------------------------------------

    hazardous_keywords = [
        "battery",
        "fluorescent",
        "thermometer",
        "hazardous"
    ]

    for keyword in hazardous_keywords:
        if keyword in name:
            return "Hazardous_waste"


    return None


# ============================================================
# IMAGE EXTENSIONS
# ============================================================

IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff"
)


# ============================================================
# NORMALIZE CLASS NAME
# ============================================================

def normalize(name):

    name = str(name).strip().lower()

    name = name.replace("_", " ")
    name = name.replace("-", " ")

    return " ".join(name.split())


# ============================================================
# LOAD YAML
# ============================================================

def load_names(dataset_path):

    yaml_path = os.path.join(
        dataset_path,
        "data.yaml"
    )

    if not os.path.exists(yaml_path):

        yaml_path = os.path.join(
            dataset_path,
            "data.yml"
        )

    if not os.path.exists(yaml_path):

        print("    [ERROR] data.yaml not found")

        return {}


    try:

        with open(
            yaml_path,
            "r",
            encoding="utf-8"
        ) as f:

            data = yaml.safe_load(f)

    except Exception as e:

        print(
            f"    [ERROR] Cannot read YAML: {e}"
        )

        return {}


    names = data.get(
        "names",
        []
    )


    if isinstance(names, list):

        return {
            i: str(name)
            for i, name in enumerate(names)
        }


    if isinstance(names, dict):

        return {
            int(k): str(v)
            for k, v in names.items()
        }


    return {}


# ============================================================
# FIND SPLIT
# ============================================================

def find_split(dataset_path, split):

    possible = []

    if split == "train":

        possible = [
            "train"
        ]

    elif split == "valid":

        possible = [
            "valid",
            "val"
        ]

    elif split == "test":

        possible = [
            "test"
        ]


    for name in possible:

        path = os.path.join(
            dataset_path,
            name
        )

        if os.path.isdir(path):

            return path


    return None


# ============================================================
# POLYGON → BOUNDING BOX
# ============================================================

def polygon_to_bbox(coords):

    if len(coords) < 6:

        return None


    if len(coords) % 2 != 0:

        return None


    xs = coords[0::2]
    ys = coords[1::2]


    xmin = min(xs)
    xmax = max(xs)

    ymin = min(ys)
    ymax = max(ys)


    width = xmax - xmin
    height = ymax - ymin


    if width <= 0 or height <= 0:

        return None


    xcenter = (
        xmin + xmax
    ) / 2

    ycenter = (
        ymin + ymax
    ) / 2


    return (
        xcenter,
        ycenter,
        width,
        height
    )


# ============================================================
# GET TARGET CLASS
# ============================================================

def get_target_class(
    source_name,
    dataset_name,
    config
):

    normalized = normalize(
        source_name
    )


    # --------------------------------------------------------
    # 1. DATASET-SPECIFIC MAPPING
    # --------------------------------------------------------

    if config is not None:

        for source_class, target_class in config.get(
            "mapping",
            {}
        ).items():

            if normalize(source_class) == normalized:

                return target_class


        default = config.get(
            "default"
        )

        if default is not None:

            return default


    # --------------------------------------------------------
    # 2. AUTOMATIC MAPPING
    # --------------------------------------------------------

    return auto_map_class(
        source_name
    )


# ============================================================
# CONVERT ONE LABEL
# ============================================================

def convert_label(
    line,
    source_names,
    config,
    dataset_name
):

    parts = line.strip().split()


    if len(parts) < 5:

        return None


    # --------------------------------------------------------
    # SOURCE CLASS
    # --------------------------------------------------------

    try:

        source_id = int(
            float(parts[0])
        )

    except Exception:

        return None


    if source_id not in source_names:

        return None


    source_name = source_names[
        source_id
    ]


    # --------------------------------------------------------
    # TARGET CLASS
    # --------------------------------------------------------

    target = get_target_class(
        source_name,
        dataset_name,
        config
    )


    if target is None:

        return None


    target_id = TARGET_IDS[
        target
    ]


    # --------------------------------------------------------
    # COORDINATES
    # --------------------------------------------------------

    try:

        coords = [
            float(x)
            for x in parts[1:]
        ]

    except Exception:

        return None


    # --------------------------------------------------------
    # DETECTION FORMAT
    # --------------------------------------------------------

    if len(coords) == 4:

        xcenter = coords[0]
        ycenter = coords[1]

        width = coords[2]
        height = coords[3]


    # --------------------------------------------------------
    # SEGMENTATION FORMAT
    # --------------------------------------------------------

    elif len(coords) >= 6:

        bbox = polygon_to_bbox(
            coords
        )

        if bbox is None:

            return None


        (
            xcenter,
            ycenter,
            width,
            height
        ) = bbox


    else:

        return None


    # --------------------------------------------------------
    # CLAMP VALUES
    # --------------------------------------------------------

    xcenter = max(
        0,
        min(1, xcenter)
    )

    ycenter = max(
        0,
        min(1, ycenter)
    )

    width = max(
        0,
        min(1, width)
    )

    height = max(
        0,
        min(1, height)
    )


    if width <= 0 or height <= 0:

        return None


    return (
        f"{target_id} "
        f"{xcenter:.6f} "
        f"{ycenter:.6f} "
        f"{width:.6f} "
        f"{height:.6f}"
    )


# ============================================================
# CLEAR OUTPUT
# ============================================================

def clear_output():

    if os.path.exists(
        OUTPUT_ROOT
    ):

        print(
            "\nRemoving old merged dataset..."
        )

        shutil.rmtree(
            OUTPUT_ROOT
        )


    os.makedirs(
        OUTPUT_ROOT
    )


# ============================================================
# PROCESS DATASET
# ============================================================

def process_dataset(
    dataset_name,
    config
):

    print("\n")
    print("=" * 70)
    print(dataset_name)
    print("=" * 70)


    dataset_path = os.path.join(
        DATASET_ROOT,
        dataset_name
    )


    if not os.path.isdir(
        dataset_path
    ):

        print(
            "[ERROR] Dataset not found:"
        )

        print(
            dataset_path
        )

        return


    print(
        "[OK] Dataset found"
    )


    # --------------------------------------------------------
    # LOAD CLASS NAMES
    # --------------------------------------------------------

    source_names = load_names(
        dataset_path
    )


    print(
        f"[OK] {len(source_names)} source classes found"
    )


    if not source_names:

        print(
            "[ERROR] No classes found"
        )

        return


    # --------------------------------------------------------
    # PRINT MAPPING
    # --------------------------------------------------------

    print("\nClass mapping:")


    for source_id, source_name in source_names.items():

        target = get_target_class(
            source_name,
            dataset_name,
            config
        )


        if target is not None:

            print(
                f"  {source_id:3} : "
                f"{source_name:<40} "
                f"-> {target}"
            )

        else:

            print(
                f"  {source_id:3} : "
                f"{source_name:<40} "
                f"-> IGNORE"
            )


    # --------------------------------------------------------
    # PROCESS TRAIN / VALID / TEST
    # --------------------------------------------------------

    for split in [
        "train",
        "valid",
        "test"
    ]:

        split_path = find_split(
            dataset_path,
            split
        )


        if split_path is None:

            print(
                f"\n[{split.upper()}] Not found"
            )

            continue


        images_path = os.path.join(
            split_path,
            "images"
        )

        labels_path = os.path.join(
            split_path,
            "labels"
        )


        if not os.path.isdir(
            images_path
        ):

            print(
                f"\n[{split.upper()}] images folder missing"
            )

            continue


        if not os.path.isdir(
            labels_path
        ):

            print(
                f"\n[{split.upper()}] labels folder missing"
            )

            continue


        # ----------------------------------------------------
        # OUTPUT
        # ----------------------------------------------------

        out_images = os.path.join(
            OUTPUT_ROOT,
            split,
            "images"
        )

        out_labels = os.path.join(
            OUTPUT_ROOT,
            split,
            "labels"
        )


        os.makedirs(
            out_images,
            exist_ok=True
        )

        os.makedirs(
            out_labels,
            exist_ok=True
        )


        # ----------------------------------------------------
        # GET IMAGES
        # ----------------------------------------------------

        images = []

        for filename in os.listdir(
            images_path
        ):

            if filename.lower().endswith(
                IMAGE_EXTENSIONS
            ):

                images.append(
                    filename
                )


        print(
            f"\n[{split.upper()}]"
        )

        print(
            f"Images found : {len(images)}"
        )


        copied = 0
        skipped = 0
        objects = 0


        # ----------------------------------------------------
        # PROCESS EVERY IMAGE
        # ----------------------------------------------------

        for image_filename in images:

            image_stem = os.path.splitext(
                image_filename
            )[0]


            label_filename = (
                image_stem + ".txt"
            )


            label_path = os.path.join(
                labels_path,
                label_filename
            )


            # ------------------------------------------------
            # LABEL MISSING
            # ------------------------------------------------

            if not os.path.exists(
                label_path
            ):

                print(
                    f"  [SKIP] No label: "
                    f"{image_filename}"
                )

                skipped += 1

                continue


            # ------------------------------------------------
            # READ LABEL
            # ------------------------------------------------

            try:

                with open(
                    label_path,
                    "r",
                    encoding="utf-8"
                ) as f:

                    lines = f.readlines()


            except Exception:

                print(
                    f"  [SKIP] Cannot read label: "
                    f"{label_filename}"
                )

                skipped += 1

                continue


            # ------------------------------------------------
            # CONVERT LABELS
            # ------------------------------------------------

            converted = []


            for line in lines:

                result = convert_label(
                    line,
                    source_names,
                    config,
                    dataset_name
                )


                if result is not None:

                    converted.append(
                        result
                    )


            # ------------------------------------------------
            # NO TARGET OBJECT
            # ------------------------------------------------

            if len(converted) == 0:

                skipped += 1

                continue


            # ------------------------------------------------
            # UNIQUE NAME
            # ------------------------------------------------

            dataset_prefix = (
                dataset_name
                .replace(" ", "_")
                .replace(".", "_")
                .replace("-", "_")
            )


            output_stem = (
                dataset_prefix
                + "__"
                + image_stem
            )


            extension = os.path.splitext(
                image_filename
            )[1]


            output_image = (
                output_stem
                + extension
            )


            output_label = (
                output_stem
                + ".txt"
            )


            # ------------------------------------------------
            # COPY IMAGE
            # ------------------------------------------------

            shutil.copy2(
                os.path.join(
                    images_path,
                    image_filename
                ),
                os.path.join(
                    out_images,
                    output_image
                )
            )


            # ------------------------------------------------
            # WRITE LABEL
            # ------------------------------------------------

            with open(
                os.path.join(
                    out_labels,
                    output_label
                ),
                "w",
                encoding="utf-8"
            ) as f:

                f.write(
                    "\n".join(
                        converted
                    )
                )

                f.write("\n")


            copied += 1

            objects += len(
                converted
            )


        print(
            f"Copied      : {copied}"
        )

        print(
            f"Skipped     : {skipped}"
        )

        print(
            f"Objects     : {objects}"
        )


# ============================================================
# DISCOVER DATASETS
# ============================================================

def discover_datasets():

    datasets = []


    if not os.path.isdir(
        DATASET_ROOT
    ):

        return datasets


    for name in sorted(
        os.listdir(DATASET_ROOT)
    ):

        path = os.path.join(
            DATASET_ROOT,
            name
        )


        if not os.path.isdir(
            path
        ):

            continue


        # Ignore merged output folder
        if name == "Final_merged_dataset":

            continue


        datasets.append(
            name
        )


    return datasets


# ============================================================
# CREATE DATA.YAML
# ============================================================

def create_yaml():

    data = {

        "path": os.path.abspath(
            OUTPUT_ROOT
        ),

        "train": "train/images",

        "val": "valid/images",

        "test": "test/images",

        "nc": len(
            TARGET_CLASSES
        ),

        "names": TARGET_CLASSES
    }


    yaml_path = os.path.join(
        OUTPUT_ROOT,
        "data.yaml"
    )


    with open(
        yaml_path,
        "w",
        encoding="utf-8"
    ) as f:

        yaml.safe_dump(
            data,
            f,
            sort_keys=False
        )


    return yaml_path


# ============================================================
# FINAL COUNT
# ============================================================

def final_count():

    print("\n")
    print("=" * 70)
    print("FINAL DATASET COUNT")
    print("=" * 70)


    total_images = 0
    total_labels = 0


    for split in [
        "train",
        "valid",
        "test"
    ]:

        images_path = os.path.join(
            OUTPUT_ROOT,
            split,
            "images"
        )

        labels_path = os.path.join(
            OUTPUT_ROOT,
            split,
            "labels"
        )


        image_count = 0
        label_count = 0


        if os.path.isdir(
            images_path
        ):

            image_count = len([
                f
                for f in os.listdir(
                    images_path
                )
                if f.lower().endswith(
                    IMAGE_EXTENSIONS
                )
            ])


        if os.path.isdir(
            labels_path
        ):

            label_count = len([
                f
                for f in os.listdir(
                    labels_path
                )
                if f.endswith(
                    ".txt"
                )
            ])


        total_images += image_count
        total_labels += label_count


        print(
            f"{split.upper():8} "
            f"Images = {image_count:6} "
            f"Labels = {label_count:6}"
        )


    print("-" * 70)


    print(
        f"TOTAL    "
        f"Images = {total_images:6} "
        f"Labels = {total_labels:6}"
    )


# ============================================================
# FINAL CLASS OBJECT COUNT
# ============================================================

def final_class_count():

    print("\n")
    print("=" * 70)
    print("FINAL CLASS OBJECT COUNT")
    print("=" * 70)


    counts = {
        class_name: 0
        for class_name in TARGET_CLASSES
    }


    for split in [
        "train",
        "valid",
        "test"
    ]:

        labels_path = os.path.join(
            OUTPUT_ROOT,
            split,
            "labels"
        )


        if not os.path.isdir(
            labels_path
        ):

            continue


        for filename in os.listdir(
            labels_path
        ):

            if not filename.endswith(
                ".txt"
            ):

                continue


            path = os.path.join(
                labels_path,
                filename
            )


            try:

                with open(
                    path,
                    "r",
                    encoding="utf-8"
                ) as f:

                    for line in f:

                        parts = line.strip().split()


                        if not parts:

                            continue


                        try:

                            class_id = int(
                                float(parts[0])
                            )

                        except Exception:

                            continue


                        if (
                            0 <= class_id
                            < len(TARGET_CLASSES)
                        ):

                            counts[
                                TARGET_CLASSES[
                                    class_id
                                ]
                            ] += 1


            except Exception:

                continue


    for class_name in TARGET_CLASSES:

        class_id = TARGET_IDS[
            class_name
        ]

        print(
            f"{class_id} - "
            f"{class_name:<18} : "
            f"{counts[class_name]:8} objects"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print("SMART WASTE DATASET MERGER")
    print("=" * 70)


    print("\nProject:")
    print(PROJECT_DIR)


    print("\nDataset root:")
    print(DATASET_ROOT)


    print("\nOutput:")
    print(OUTPUT_ROOT)


    print("\nFinal classes:")

    for i, name in enumerate(
        TARGET_CLASSES
    ):

        print(
            f"  {i} = {name}"
        )


    print("=" * 70)


    # --------------------------------------------------------
    # CHECK ROOT
    # --------------------------------------------------------

    if not os.path.isdir(
        DATASET_ROOT
    ):

        print(
            "\n[ERROR]"
        )

        print(
            "Dataset folder does not exist!"
        )

        return


    # --------------------------------------------------------
    # CLEAR OLD DATA
    # --------------------------------------------------------

    clear_output()


    # --------------------------------------------------------
    # DISCOVER ALL DATASETS
    # --------------------------------------------------------

    datasets = discover_datasets()


    print("\nDatasets discovered:")

    for dataset_name in datasets:

        print(
            f"  [FOUND] {dataset_name}"
        )


    print(
        f"\nTotal datasets discovered: "
        f"{len(datasets)}"
    )


    # --------------------------------------------------------
    # PROCESS ALL DATASETS
    # --------------------------------------------------------

    for dataset_name in datasets:

        config = DATASET_CONFIG.get(
            dataset_name
        )


        process_dataset(
            dataset_name,
            config
        )


    # --------------------------------------------------------
    # YAML
    # --------------------------------------------------------

    yaml_path = create_yaml()


    # --------------------------------------------------------
    # COUNTS
    # --------------------------------------------------------

    final_count()

    final_class_count()


    # --------------------------------------------------------
    # DONE
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("MERGING COMPLETED")
    print("=" * 70)


    print("\nFinal dataset:")

    print(
        os.path.abspath(
            OUTPUT_ROOT
        )
    )


    print("\ndata.yaml:")

    print(
        os.path.abspath(
            yaml_path
        )
    )


    print("\nClasses:")

    for i, name in enumerate(
        TARGET_CLASSES
    ):

        print(
            f"  {i} = {name}"
        )


    print("\n")


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()