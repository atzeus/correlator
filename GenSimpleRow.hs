import Data.List

indices = ["0","1","2","3","4","5","6","7","8","9","A","B","C","D","E","F"]

setSum z x y = "sum[" ++ show (z * 8 + y) ++ "].s"++ (indices !! (x * 2)) ++ " += " ++ xr ++ "*" ++ yr ++ " + " ++ xi ++ "*" ++ yi ++ ";\n" ++
             "sum[" ++ show (z * 8 + y)  ++ "].s"++ ( indices !! (x * 2 + 1)) ++ " += " ++ xi ++ "*" ++ yr ++ " + " ++ xr ++ "*" ++ yi ++ ";\n"
  where xr = "rowmem[time].s" ++ indices !! (x * 2)
        xi = "rowmem[time].s" ++ indices !! (x * 2 + 1)
        yr = "memy" ++ show z ++ ".s" ++ indices !! (y * 2)
        yi = "memy" ++ show z ++ ".s" ++ indices !! (y * 2 + 1)
        
setSums = intercalate "\n" [setSum z x y | z <- [0,1], y <- [0..7],  x <- [0..7]]
