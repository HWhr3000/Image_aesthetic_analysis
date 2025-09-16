import pandas as pd
from aesthetics_eval.postprocess import map_and_flag

def test_map_and_flag_minimal(tmp_path):
    df = pd.DataFrame([{
        "img_file":"x.jpg",
        "Image width (pixels)":512,
        "Image height (pixels)":512,
        "Aspect ratio":1.0,
        "RMS contrast":22.0,
        "Edge density":600.0,
        "mean L channel":90.0,
        "mean a channel":0.0,
        "mean b channel (Lab)":10.0,
        "Color entropy":5.0,
        "Mirror symmetry":4.0,
        "Balance":15.0,
        "Fourier slope":-2.5,
        "1st-order EOE":4.2,
        "2nd-order EOE":4.4
    }])
    csv = tmp_path/"x.csv"; df.to_csv(csv, index=False)
    thresholds = {
      "lab": {"L_min":88,"L_max":92,"a_min":-1,"a_max":3,"b_min":8,"b_max":12},
      "rms_contrast":{"min":15,"max":30,"target":22},
      "sharpness":{"edge_density_min":500,"eoe_min":4},
      "color_entropy":{"min":4,"max":6},
      "fourier_slope":{"min":-3.0,"max":-2.0},
      "symmetry_max":0.05,
      "balance_max":0.20
    }
    out_df, rows = map_and_flag(str(csv), thresholds)
    r = rows[0]
    assert r["compliance"]["lab_ok"]
    assert r["compliance"]["rms_ok"]
    assert r["compliance"]["edge_ok"]
    assert r["compliance"]["eoe_ok"]
    assert r["compliance"]["color_entropy_ok"]
    assert r["compliance"]["fourier_ok"]
    assert r["compliance"]["symmetry_ok"]
    assert r["compliance"]["balance_ok"]
    assert r["compliance"]["compliance_pct"] == 100.0
