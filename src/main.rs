// Be sure to set the environmental variables LIBTORCH and LD_LIBRARY_PATH to the libtorch install path

use std::env;

use tch::{vision::image, CModule};

fn main() {
    // arg 1, as arg 0 is the application's path
    let image_path = match env::args().nth(1) {
        Some(val) => val,
        None => {
            println!("Please pass image path as a command line argument");
            return;
        }
    };

    let img = image::load_and_resize(&image_path, 224, 224)
        .expect("Error opening image")
        .unsqueeze(0)
        / 255;

    let model = CModule::load("src/traced_model.pt").expect("Error loading model");

    let preds = model
        .forward_ts(&[img])
        .expect("Error making predictions")
        .softmax(-1, None)
        .get(0);

    let mut top_class = "None";
    let mut top_class_prob = 0.0;

    for i in 0..preds.size1().expect("Size error") {
        let prob = 100.0 * preds.double_value(&[i]);
        if prob > top_class_prob {
            top_class_prob = prob;
            top_class = CLASSES[i as usize];
        }
    }

    println!("{top_class}: {top_class_prob}%");

    return;
}

const CLASSES: [&str; 37] = [
    "Abyssinian",
    "Bengal",
    "Birman",
    "Bombay",
    "British_Shorthair",
    "Egyptian_Mau",
    "Maine_Coon",
    "Persian",
    "Ragdoll",
    "Russian_Blue",
    "Siamese",
    "Sphynx",
    "american_bulldog",
    "american_pit_bull_terrier",
    "basset_hound",
    "beagle",
    "boxer",
    "chihuahua",
    "english_cocker_spaniel",
    "english_setter",
    "german_shorthaired",
    "great_pyrenees",
    "havanese",
    "japanese_chin",
    "keeshond",
    "leonberger",
    "miniature_pinscher",
    "newfoundland",
    "pomeranian",
    "pug",
    "saint_bernard",
    "samoyed",
    "scottish_terrier",
    "shiba_inu",
    "staffordshire_bull_terrier",
    "wheaten_terrier",
    "yorkshire_terrier",
];
