"""
Exploring the College Scorecard MERGED files.

1. Print the column names in a MERGED file
2. Loop through every MERGEDYYYY_YY_PP.csv, pull out Northeastern University's rows
   (all campuses: Boston, Professional Programs, Oakland)
3. Plot Northeastern's mean salary vs. the mean salary of all college grads, year by year
"""

import glob
import os

import matplotlib.pyplot as plt
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw_data_files")


# Every Northeastern University campus starts with this name:
#   "Northeastern University"                        (Boston, UNITID 167358)
#   "Northeastern University Professional Programs"  (Boston, UNITID 482705)
#   "Northeastern University Oakland"                (UNITID 118888, formerly Mills College)
# Matching the start of the name avoids "Northeastern Illinois University", etc.
NEU_PREFIX = "Northeastern University"

# The Scorecard uses these strings to mean "no data"
NA_VALUES = ["NULL", "PrivacySuppressed", "PS", "NA"]

# MN_EARN_WNE_P10  = mean earnings of students working and not enrolled, 10 yrs after entry
# COUNT_WNE_P10    = number of students that mean is based on (used for weighting)
EARN_COL = "MN_EARN_WNE_P10"
COUNT_COL = "COUNT_WNE_P10"

# US News 2025 Best National Universities (published Sept 2024), ranks 1-46.
# Ties mean "top 20" is exactly 20 schools and "top 50" is 49 (the next tie is at #51).
# Format: UNITID: (rank, name). Written from memory, so check it against the US News
# site and edit any ranks that are wrong.
US_NEWS_2025 = {
    186131: (1, "Princeton University"),
    166683: (2, "Massachusetts Institute of Technology"),
    166027: (3, "Harvard University"),
    243744: (4, "Stanford University"),
    130794: (5, "Yale University"),
    110404: (6, "California Institute of Technology"),
    198419: (6, "Duke University"),
    162928: (6, "Johns Hopkins University"),
    147767: (6, "Northwestern University"),
    215062: (10, "University of Pennsylvania"),
    190415: (11, "Cornell University"),
    144050: (11, "University of Chicago"),
    217156: (13, "Brown University"),
    190150: (13, "Columbia University"),
    182670: (15, "Dartmouth College"),
    110662: (15, "UCLA"),
    110635: (17, "UC Berkeley"),
    227757: (18, "Rice University"),
    152080: (18, "University of Notre Dame"),
    221999: (18, "Vanderbilt University"),
    211440: (21, "Carnegie Mellon University"),
    170976: (21, "University of Michigan-Ann Arbor"),
    179867: (21, "Washington University in St. Louis"),
    139658: (24, "Emory University"),
    131496: (24, "Georgetown University"),
    234076: (24, "University of Virginia"),
    199120: (27, "UNC Chapel Hill"),
    123961: (27, "University of Southern California"),
    110680: (29, "UC San Diego"),
    193900: (30, "New York University"),
    134130: (30, "University of Florida"),
    228778: (30, "UT Austin"),
    139755: (33, "Georgia Tech"),
    110644: (33, "UC Davis"),
    110653: (33, "UC Irvine"),
    145637: (33, "UIUC"),
    164924: (37, "Boston College"),
    168148: (37, "Tufts University"),
    110705: (39, "UC Santa Barbara"),
    240444: (39, "UW-Madison"),
    164988: (41, "Boston University"),
    204796: (41, "Ohio State University"),
    186380: (41, "Rutgers-New Brunswick"),
    163286: (44, "University of Maryland-College Park"),
    195030: (44, "University of Rochester"),
    213543: (46, "Lehigh University"),
    243780: (46, "Purdue University"),
    139959: (46, "University of Georgia"),
    236948: (46, "University of Washington"),
}
TOP_20_IDS = [uid for uid, (rank, _) in US_NEWS_2025.items() if rank <= 20]
TOP_50_IDS = [uid for uid, (rank, _) in US_NEWS_2025.items() if rank <= 50]


def get_columns(filename):
    """Return the list of column names in a CSV (reads the header only)."""
    return pd.read_csv(filename, nrows=0).columns.tolist()


def year_from_filename(filename):
    """'MERGED2013_14_PP.csv' -> 2013"""
    return int(os.path.basename(filename)[6:10])


def load_earnings(filename):
    """Read only the columns we need from one MERGED file and make them numeric."""
    df = pd.read_csv(filename, usecols=["UNITID", "OPEID6", "INSTNM", EARN_COL, COUNT_COL],
                     na_values=NA_VALUES, low_memory=False)
    df[EARN_COL] = pd.to_numeric(df[EARN_COL], errors="coerce")
    df[COUNT_COL] = pd.to_numeric(df[COUNT_COL], errors="coerce")
    return df


def dedupe_earnings(df):
    """
    Earnings are reported per OPEID (a financial-aid ID), not per campus, so
    branch campuses that share an OPEID get an identical copy of the same numbers.
    (e.g. NEU Boston and NEU Professional Programs both show 4,963 grads at $69,500.)
    Keep one copy so the same graduates aren't counted twice.
    """
    has_data = df.dropna(subset=[EARN_COL, COUNT_COL])
    return has_data.drop_duplicates(subset=["OPEID6", EARN_COL, COUNT_COL])


def weighted_mean(df):
    """Mean salary weighted by how many graduates each row covers."""
    if df[COUNT_COL].sum() == 0:
        return None
    return (df[EARN_COL] * df[COUNT_COL]).sum() / df[COUNT_COL].sum()


# ===================== Computer science & business only =====================
# The MERGED files don't split earnings by major, so this part uses the
# FieldOfStudyData files instead (one row per school + program + degree level).
#   - They only have MEDIAN earnings, measured after GRADUATING (not after entry).
#   - Only 3 files (1819_1920, 1920_2021, 2122_2223) have any earnings filled in.
#   - The data is one summary number per program, not individual grads, so we pick
#     how many years after graduating to look at:
#         YEARS_OUT = 1  -> in all 3 files
#         YEARS_OUT = 4  -> 1819_1920, 2122_2223
#         YEARS_OUT = 5  -> 1920_2021, 2122_2223
#         YEARS_OUT = 7  -> 2122_2223 only
#   - Each file pools two graduating classes. For later YEARS_OUT the grads are from an
#     earlier class than the file name (they need time to reach 5 yrs out).
YEARS_OUT = 5                                    # earnings measured in the 5th year after graduating
FOS_EARN_COL = f"EARN_MDN_{YEARS_OUT}YR"         # median earnings N years after graduating
FOS_COUNT_COL = f"EARN_COUNT_WNE_{YEARS_OUT}YR"  # grads that median is based on
BACHELORS = 3                         # CREDLEV 3 = bachelor's degree

# CIP codes are the federal major codes. 11xx = Computer & Information Sciences,
# 52xx = Business, Management, Marketing (finance, accounting, marketing, intl business, ...)
MAJORS = {"Computer Science": "11", "Business": "52"}


def load_fos(filename):
    """Read one FieldOfStudyData file: bachelor's degrees in CS or business only."""
    df = pd.read_csv(filename,
                     usecols=["UNITID", "OPEID6", "INSTNM", "CIPCODE", "CIPDESC", "CREDLEV",
                              FOS_EARN_COL, FOS_COUNT_COL],
                     na_values=NA_VALUES, dtype={"CIPCODE": str}, low_memory=False)
    df[FOS_EARN_COL] = pd.to_numeric(df[FOS_EARN_COL], errors="coerce")
    df[FOS_COUNT_COL] = pd.to_numeric(df[FOS_COUNT_COL], errors="coerce")

    df = df[df["CREDLEV"] == BACHELORS].copy()
    df["MAJOR"] = None
    for major, prefix in MAJORS.items():
        df.loc[df["CIPCODE"].str.startswith(prefix, na=False), "MAJOR"] = major
    df = df.dropna(subset=["MAJOR", FOS_EARN_COL, FOS_COUNT_COL])

    # Same issue as MERGED: campuses sharing an OPEID get copies of the same numbers
    return df.drop_duplicates(subset=["OPEID6", "CIPCODE", FOS_EARN_COL, FOS_COUNT_COL])


def fos_weighted_avg(df):
    """Average of program medians, weighted by # of grads in each program."""
    if len(df) == 0:
        return None
    return (df[FOS_EARN_COL] * df[FOS_COUNT_COL]).sum() / df[FOS_COUNT_COL].sum()


def by_major_summary():
    """One row per (file, major) with NEU / top 20 / top 50 / all-colleges earnings."""
    rows = []
    for f in sorted(glob.glob(os.path.join(DATA_DIR, "FieldOfStudyData*_PP.csv"))):
        df = load_fos(f)
        if len(df) == 0:   # 6 of the 9 files have no earnings data
            continue
        cohort = os.path.basename(f)[16:25]   # e.g. "1819_1920"

        for major in MAJORS:
            m = df[df["MAJOR"] == major]
            neu = m[m["INSTNM"].str.startswith(NEU_PREFIX, na=False)]
            rows.append({"COHORT": cohort, "MAJOR": major,
                         "NEU": fos_weighted_avg(neu),
                         "TOP20": fos_weighted_avg(m[m["UNITID"].isin(TOP_20_IDS)]),
                         "TOP50": fos_weighted_avg(m[m["UNITID"].isin(TOP_50_IDS)]),
                         "ALL": fos_weighted_avg(m)})
    return pd.DataFrame(rows)


def plot_by_major(summary):
    """One panel per major: Northeastern vs. top 20, top 50, and all colleges."""
    fig, axes = plt.subplots(1, len(MAJORS), figsize=(12, 5), sharey=True)
    for ax, major in zip(axes, MAJORS):
        s = summary[summary["MAJOR"] == major]
        ax.plot(s["COHORT"], s["TOP20"], marker="o", label="US News top 20 (2025)")
        ax.plot(s["COHORT"], s["TOP50"], marker="o", label="US News top 50 (2025)")
        ax.plot(s["COHORT"], s["NEU"], marker="o", label="Northeastern University")
        ax.plot(s["COHORT"], s["ALL"], marker="o", label="All colleges")
        ax.set_title(f"{major} bachelor's")
        ax.set_xlabel("Graduating classes (file name)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel(f"Median earnings {YEARS_OUT} yrs after graduating ($)")
    axes[0].legend()
    fig.suptitle(f"Earnings {YEARS_OUT} years after graduating, by major\n"
                 "(grad-weighted average of program medians)")
    fig.tight_layout()


def main():
    files = sorted(glob.glob(os.path.join(DATA_DIR, "MERGED*_PP.csv")))

    # ---- 1. Column names ----
    cols = get_columns(files[-1])
    print(f"{len(cols)} columns in {os.path.basename(files[-1])}")
    print(cols[:50], "...")   # print(cols) to see all of them

    # ---- 2. Loop through every year ----
    rows = []
    neu_frames = []
    for f in files:
        df = load_earnings(f)
        year = year_from_filename(f)

        neu = df[df["INSTNM"].str.startswith(NEU_PREFIX, na=False)].copy()
        neu["YEAR"] = year
        neu_frames.append(neu)

        neu_clean = dedupe_earnings(neu)
        all_clean = dedupe_earnings(df)

        top20 = all_clean[all_clean["UNITID"].isin(TOP_20_IDS)]
        top50 = all_clean[all_clean["UNITID"].isin(TOP_50_IDS)]

        rows.append({"YEAR": year,
                     "NEU_MEAN": weighted_mean(neu_clean),     # all NEU campuses combined
                     "NATL_MEAN": weighted_mean(all_clean),    # every college in the file
                     "TOP20_MEAN": weighted_mean(top20),       # US News top 20
                     "TOP50_MEAN": weighted_mean(top50)})      # US News top 50

    # Northeastern's rows, every year, every campus -- use this to explore other columns
    neu_all_years = pd.concat(neu_frames)
    print(neu_all_years.dropna(subset=[EARN_COL])
          [["YEAR", "INSTNM", EARN_COL, COUNT_COL]].to_string(index=False))

    summary = pd.DataFrame(rows).dropna()   # keep only years that have earnings data
    print(summary.to_string(index=False))

    # ---- 3. Plot ----
    plt.figure(figsize=(9, 5))
    plt.plot(summary["YEAR"], summary["NEU_MEAN"], marker="o",
             label="Northeastern University (all campuses)")
    plt.plot(summary["YEAR"], summary["NATL_MEAN"], marker="o",
             label="All colleges")
    plt.title("Mean earnings 10 years after entry (weighted by # of grads)")
    plt.xlabel("Data year (start of academic year in file name)")
    plt.ylabel("Mean earnings ($)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    # ---- 4. Plot: Northeastern vs. US News top 20 / top 50 ----
    plt.figure(figsize=(9, 5))
    plt.plot(summary["YEAR"], summary["TOP20_MEAN"], marker="o",
             label="US News top 20 (2025)")
    plt.plot(summary["YEAR"], summary["TOP50_MEAN"], marker="o",
             label="US News top 50 (2025)")
    plt.plot(summary["YEAR"], summary["NEU_MEAN"], marker="o",
             label="Northeastern University (all campuses)")
    plt.title("Mean earnings 10 years after entry: Northeastern vs. top-ranked schools")
    plt.xlabel("Data year (start of academic year in file name)")
    plt.ylabel("Mean earnings ($)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    # ---- 5. Computer science & business only (FieldOfStudyData files) ----
    major_summary = by_major_summary()
    print(major_summary.round(0).to_string(index=False))
    plot_by_major(major_summary)

    plt.show()   # shows all figures


if __name__ == "__main__":
    main()
