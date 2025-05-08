import React from "react";
import { Box, Typography, Button } from "@mui/material";
import CheckCircleRoundedIcon from "@mui/icons-material/CheckCircleRounded";
import EastRoundedIcon from "@mui/icons-material/EastRounded";

const SuccessPage = () => {
    const handleContinue = () => {
        console.log("Continuing to the next page...");
    };

    return (
        <Box
            sx={{
                position: "absolute",
                top: 0,
                left: 0,
                width: "100vw",
                height: "100vh",
                bgcolor: "#fbfbfe",
                display: "flex",
                justifyContent: "center",
                alignItems: "center",
            }}
        >
            <Box
                sx={{
                    p: 4,
                    borderRadius: 3,
                    width: "100%",
                    maxWidth: 400,
                    textAlign: "center",
                }}
            >
                <CheckCircleRoundedIcon
                    sx={{
                        fontSize: 50,
                        color: "#2f27ce",
                        mb: 4,
                    }}
                />

                <Typography
                    variant="body1"
                    sx={{
                        color: "#050315",
                        mb: 2,
                    }}
                >
                    You have successfully logged in!
                </Typography>

                <Button
                    variant="contained"
                    endIcon={<EastRoundedIcon />}
                    onClick={handleContinue}
                    sx={{
                        backgroundColor: "#2f27ce",
                        color: "#fbfbfe",
                        borderRadius: "30px",
                        fontWeight: "bold",
                        textTransform: "none",
                        fontSize: "0.875rem", // Smaller font size
                        px: 3, // Reduced padding
                        py: 1, // Reduced padding
                        "&:hover": {
                            backgroundColor: "#433bff",
                        },
                    }}
                    fullWidth
                >
                    Continue
                </Button>
            </Box>
        </Box>
    );
};

export default SuccessPage;
