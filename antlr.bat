SET classpath=.;C:\projects\antlr\antlr-4.9-complete.jar;%classpath%
SET ANTLR_TOOL=java org.antlr.v4.Tool
IF "%~1" == "" (%ANTLR_TOOL%  PS.g4) ELSE (%ANTLR_TOOL% %*)
