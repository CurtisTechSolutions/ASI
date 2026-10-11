//! Text, audio, images and video as symbols, and networks taught from them.

use latticefsm::gzip;
use latticefsm::machine::Settings;
use latticefsm::media::{
    cells, diff_symbols, image_symbols, make_dataset, order, read_image, read_image_bytes, read_labels, text_symbols,
    verdict, video_symbols,
};
use latticefsm::network::{load_network, Credit, Merge, Network, Vote};

fn tmp(name: &str) -> std::path::PathBuf {
    let d = std::env::temp_dir().join(format!("latticefsm-media-test-{}-{name}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// A PNG of RGB rows, each row with the next of the five filters.
fn png(width: usize, height: usize, rgb: &[u8]) -> Vec<u8> {
    let stride = width * 3;
    let mut raw = Vec::new();
    let mut prev = vec![0u8; stride];
    for y in 0..height {
        let line = &rgb[y * stride..(y + 1) * stride];
        let f = (y % 5) as u8;
        raw.push(f);
        for i in 0..stride {
            let a = if i >= 3 { line[i - 3] as i32 } else { 0 };
            let b = prev[i] as i32;
            let c = if i >= 3 { prev[i - 3] as i32 } else { 0 };
            let pred = match f {
                0 => 0,
                1 => a,
                2 => b,
                3 => (a + b) / 2,
                _ => {
                    let p = a + b - c;
                    let (pa, pb, pc) = ((p - a).abs(), (p - b).abs(), (p - c).abs());
                    if pa <= pb && pa <= pc {
                        a
                    } else if pb <= pc {
                        b
                    } else {
                        c
                    }
                }
            };
            raw.push(((line[i] as i32 - pred) & 255) as u8);
        }
        prev = line.to_vec();
    }
    let gz = gzip::compress(&raw);
    let mut z = vec![0x78, 0x9c];
    z.extend_from_slice(&gz[10..gz.len() - 8]);
    z.extend_from_slice(&[0, 0, 0, 0]);
    let chunk = |tag: &[u8], body: &[u8]| {
        let mut c = (body.len() as u32).to_be_bytes().to_vec();
        c.extend_from_slice(tag);
        c.extend_from_slice(body);
        c.extend_from_slice(&[0, 0, 0, 0]);
        c
    };
    let mut ihdr = (width as u32).to_be_bytes().to_vec();
    ihdr.extend_from_slice(&(height as u32).to_be_bytes());
    ihdr.extend_from_slice(&[8, 2, 0, 0, 0]);
    let mut out = b"\x89PNG\r\n\x1a\n".to_vec();
    out.extend(chunk(b"IHDR", &ihdr));
    out.extend(chunk(b"IDAT", &z));
    out.extend(chunk(b"IEND", &[]));
    out
}

#[test]
fn a_png_and_a_ppm_of_one_picture_read_alike() {
    let (w, h) = (13usize, 11usize);
    let rgb: Vec<u8> = (0..w * h * 3)
        .map(|i| ((i % 3) * 50 + (i / 3 % w) * 19 + (i / 3 / w) * 7) as u8)
        .collect();
    let a = read_image_bytes(&png(w, h, &rgb)).unwrap();
    let mut ppm = format!("P6\n# a comment\n{w} {h}\n255\n").into_bytes();
    ppm.extend_from_slice(&rgb);
    let b = read_image_bytes(&ppm).unwrap();
    assert_eq!((a.width, a.height), (w, h));
    assert_eq!(a.px, b.px);
    assert!(read_image_bytes(b"GIF89a").is_err());
}

#[test]
fn cells_orders_and_differences() {
    assert_eq!(order(3, "center-out").unwrap()[0], 4);
    assert_eq!(order(4, "center-out").unwrap()[..4], [5, 6, 9, 10]);
    let img = latticefsm::media::Image {
        width: 4,
        height: 4,
        px: [vec![0; 8], vec![255; 8]].concat(),
    };
    assert_eq!(cells(&img, 2), vec![0, 0, 255, 255]);
    assert_eq!(
        image_symbols(&[0, 63, 64, 255], 4, "rows").unwrap(),
        ["L0", "L0", "L1", "L3"]
    );
    assert_eq!(
        diff_symbols(&[100; 4], &[107, 108, 52, 151], "rows").unwrap(),
        ["D0", "D+1", "D-2", "D+2"]
    );
    assert_eq!(text_symbols("Hi, Bob!\t2").concat(), "hi,_bob!_2");
    assert!(verdict("yes").unwrap() && !verdict("0").unwrap() && verdict("maybe").is_err());
}

#[test]
fn a_video_is_keyframes_differences_and_its_sound() {
    let d = tmp("video");
    let labels = make_dataset("video", &d.to_string_lossy(), 2, 1).unwrap();
    let rows = read_labels(&labels, true).unwrap();
    let v = video_symbols(&rows[0].0, 8, 4, "rows", 0, 16, true, 8).unwrap();
    assert_eq!(v.iter().filter(|s| *s == "|").count(), 8);
    assert!(v[1..65].iter().all(|s| s.starts_with('L')));
    assert!(v.iter().any(|s| s.starts_with('q')));
    let mut net = Network::from_architecture(
        "video>3",
        &["a".to_string()],
        1,
        Merge::Concat,
        Vote::Majority,
        &Settings::default(),
    )
    .unwrap();
    net.compile(Some(&[0]), 0.8, 1.0, 1.0, Credit::Shared).unwrap();
    let data: Vec<(String, bool)> = rows.iter().map(|(x, y)| (x.clone(), verdict(y).unwrap())).collect();
    net.fit(&data, 2).unwrap();
    let path = d.join("n.json");
    net.save(path.to_str().unwrap()).unwrap();
    let mut back = load_network(path.to_str().unwrap()).unwrap();
    assert_eq!(back.to_json().dump(), net.to_json().dump());
    let x: Vec<String> = rows.iter().map(|(x, _)| x.clone()).collect();
    assert_eq!(back.predict(&x).unwrap(), net.predict(&x).unwrap());
    std::fs::remove_dir_all(d).ok();
}

#[test]
fn images_from_files() {
    let d = tmp("images");
    let labels = make_dataset("image", &d.to_string_lossy(), 4, 1).unwrap();
    let rows = read_labels(&labels, true).unwrap();
    let img = read_image(&rows[0].0).unwrap();
    assert_eq!((img.width, img.height), (32, 32));
    let mut net = Network::from_architecture(
        "image>3",
        &["a".to_string()],
        1,
        Merge::Concat,
        Vote::Majority,
        &Settings::default(),
    )
    .unwrap();
    assert_eq!(net.tokenize(&rows[0].0).unwrap().len(), 64);
    std::fs::remove_dir_all(d).ok();
}
