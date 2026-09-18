-- Population Stability Index, computed in DuckDB.
--
-- Reference: training months 0-4. Numeric bins are the reference deciles;
-- categoricals use each observed category plus an "other" bucket for anything
-- the reference never saw. Smoothing: a share below $epsilon is lifted to
-- $epsilon, so an empty bin cannot produce an infinite term. Identical
-- distributions give exactly 0, because every term is (p - p) * ln(p / p).
--
-- The caller registers five relations and passes $epsilon:
--   bins(feature VARCHAR, bin VARCHAR, lo DOUBLE, hi DOUBLE)  -- [lo, hi), +/-inf at the ends
--   ref_numeric(feature VARCHAR, value DOUBLE)
--   win_numeric(feature VARCHAR, value DOUBLE)
--   ref_categorical(feature VARCHAR, value VARCHAR)           -- already mapped to levels
--   win_categorical(feature VARCHAR, value VARCHAR)           -- unseen levels mapped to 'other'
--
-- Python computes only the decile edges (one pass over the reference); the
-- bucketing, the counting and the PSI arithmetic all happen here, so the SQL
-- path and the pandas path in psi.py cannot drift apart silently.

WITH ref_binned AS (
    SELECT r.feature, b.bin
    FROM ref_numeric AS r
    JOIN bins AS b
      ON b.feature = r.feature
     AND r.value >= b.lo
     AND r.value < b.hi
    UNION ALL
    SELECT feature, value AS bin FROM ref_categorical
),

win_binned AS (
    SELECT w.feature, b.bin
    FROM win_numeric AS w
    JOIN bins AS b
      ON b.feature = w.feature
     AND w.value >= b.lo
     AND w.value < b.hi
    UNION ALL
    SELECT feature, value AS bin FROM win_categorical
),

ref_counts AS (SELECT feature, bin, COUNT(*) AS n FROM ref_binned GROUP BY feature, bin),
win_counts AS (SELECT feature, bin, COUNT(*) AS n FROM win_binned GROUP BY feature, bin),

ref_totals AS (SELECT feature, SUM(n) AS total FROM ref_counts GROUP BY feature),
win_totals AS (SELECT feature, SUM(n) AS total FROM win_counts GROUP BY feature),

-- Every bin either side contributes, so a bin that appears only in the window
-- (or only in the reference) is not silently dropped.
grid AS (
    SELECT feature, bin FROM ref_counts
    UNION
    SELECT feature, bin FROM win_counts
),

shares AS (
    SELECT
        g.feature,
        g.bin,
        GREATEST(COALESCE(rc.n, 0)::DOUBLE / rt.total, $epsilon) AS p_ref,
        GREATEST(COALESCE(wc.n, 0)::DOUBLE / wt.total, $epsilon) AS p_win
    FROM grid AS g
    LEFT JOIN ref_counts AS rc ON rc.feature = g.feature AND rc.bin = g.bin
    LEFT JOIN win_counts AS wc ON wc.feature = g.feature AND wc.bin = g.bin
    JOIN ref_totals AS rt ON rt.feature = g.feature
    JOIN win_totals AS wt ON wt.feature = g.feature
)

SELECT
    feature,
    SUM((p_win - p_ref) * LN(p_win / p_ref)) AS psi,
    COUNT(*) AS n_bins
FROM shares
GROUP BY feature
ORDER BY feature;
