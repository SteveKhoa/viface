import React from "react";
import { Box, Typography, Button, Paper } from "@mui/material";
import AccountCircleRoundedIcon from "@mui/icons-material/AccountCircleRounded";

const LoginPage = () => {
    const requestToken = () => {
        console.log("Logging in with ViFace...");
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
                <Typography
                    variant="h5"
                    fontWeight="bold"
                    gutterBottom
                    sx={{ color: "#050315" }}
                >
                    Simple Ecommerce
                </Typography>

                <Typography variant="body1" sx={{ color: "#050315", mb: 4 }}>
                    Our latest biometric authentication technology.
                </Typography>

                <Button
                    variant="contained"
                    startIcon={<AccountCircleRoundedIcon />}
                    onClick={requestToken}
                    sx={{
                        backgroundColor: "#2f27ce",
                        color: "#fbfbfe",
                        fontWeight: "bold",
                        borderRadius: "30px",
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
                    Sign in with ViFace
                </Button>
            </Box>
        </Box>
    );
};

export default LoginPage;
