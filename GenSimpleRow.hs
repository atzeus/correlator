import Data.List

indices = ["0","1","2","3","4","5","6","7","8","9","A","B","C","D","E","F"]

setSum x y = "sum[ysub][" ++  show y ++ "].s"++ (indices !! (x * 2)) ++ " += " ++ xr ++ "*" ++ yr ++ " + " ++ xi ++ "*" ++ yi ++ ";\n" ++
             "sum[ysub][" ++  show y  ++ "].s"++ ( indices !! (x * 2 + 1)) ++ " += " ++ xi ++ "*" ++ yr ++ " - " ++ xr ++ "*" ++ yi ++ ";\n"
  where xr = "memx.s" ++ indices !! (x * 2)
        xi = "memx.s" ++ indices !! (x * 2 + 1)
        yr = "memy.s" ++ indices !! (y * 2)
        yi = "memy.s" ++ indices !! (y * 2 + 1)
        
setSums = intercalate "\n" [setSum x y  | y <- [0..7],  x <- [0..7]]
