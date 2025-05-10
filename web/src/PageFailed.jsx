import React from "react";
import { Box, Typography, Button } from "@mui/material";
import ErrorOutlineRoundedIcon from "@mui/icons-material/ErrorOutlineRounded";
import { STATE_LOGIN } from "./config";

const FailedPage = ({ setState, failedMsg, setFailedMsg }) => {
    const handleRetry = () => {
        setFailedMsg("");
        setState(STATE_LOGIN);
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
                fontFamily: "'Hubotsands', sans-serif",
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
                <ErrorOutlineRoundedIcon
                    sx={{
                        fontSize: 50,
                        color: "#d32f2f",
                        mb: 4,
                    }}
                />

                <Typography
                    variant="body1"
                    sx={{
                        color: "#050315",
                    }}
                >
                    Login failed. Please try again.
                </Typography>

                <Typography
                    variant="overline"
                    sx={{
                        color: "#d32f2f",
                    }}
                >
                    {failedMsg}
                </Typography>

                <Button
                    variant="contained"
                    onClick={handleRetry}
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
                    Try again.
                </Button>
            </Box>
        </Box>
    );
};

export default FailedPage;
