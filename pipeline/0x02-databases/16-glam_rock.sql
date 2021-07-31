-- SQL COMMENT
SELECT band_name, ((IFNULL(split, 2021)) - formed) AS lifespan 
FROM metal_bands
WHERE style LIKE '%glam rock%';