import React from "react";
import { Box, Typography, CircularProgress } from "@mui/material";

const WaitingPage = () => {
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
                <CircularProgress
                    size={50}
                    sx={{
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
                    Please consent your biometric to continue.
                </Typography>
            </Box>
        </Box>
    );
};

export default WaitingPage;
