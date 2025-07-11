import React, { useState } from "react";
import { Box, Typography, Button, Paper, TextField, Stack } from "@mui/material";
import AccountCircleRoundedIcon from "@mui/icons-material/AccountCircleRounded";
import PersonAddAlt1RoundedIcon from "@mui/icons-material/PersonAddAlt1Rounded";
import {
    ACCESS_TOKEN_ENDPOINT,
    STATE_FAILED,
    STATE_SUCCESS,
    STATE_WAITING,
    REGISTER_ENDPOINT,
    STATE_LOGIN,
} from "./config";

const MOCK_DOMAIN = "simpleecommerce.com";

const LoginPage = ({ setState, setFailedMsg }) => {
    const [allowSignin, setAllowSignin] = useState(false);
    const [username, setUsername] = useState("");

    const signIn = (domain, userID) => {
        (async () => {
            setState(STATE_WAITING);

            const urlWithParams = `${ACCESS_TOKEN_ENDPOINT}?domain=${domain}&user_id=${userID}`;

            const resp = await fetch(urlWithParams);
            const content = await resp.json();

            if (content.status == "200") {
                setState(STATE_SUCCESS);
                localStorage.setItem("accessToken", content.access_token);
                console.log(content.access_token);
            } else {
                setState(STATE_FAILED);
                setFailedMsg(content.msg);
            }
        })();
    };

    const register = (userID) => {
        (async () => {
            setState(STATE_WAITING);

            const urlWithParams = `${REGISTER_ENDPOINT}?user_id=${userID}`;

            await fetch(urlWithParams);
            setState(STATE_LOGIN);
        })();
    };

    const onInputChange = (event) => {
        const username = event.target.value;

        setUsername(username);

        if (event.target.value.length > 0) {
            setAllowSignin(true);
        } else {
            setAllowSignin(false);
        }
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

                <Typography variant="body1" sx={{ color: "#050315", mb: 4 }}>
                    Product: Truyen Kieu - NXB Van Hoc - 1999
                </Typography>

                <TextField
                    label="Username"
                    variant="standard"
                    fullWidth
                    sx={{
                        mb: 3,
                    }}
                    onChange={onInputChange}
                />

                <Stack spacing={2}>
                    <Button
                        variant="contained"
                        startIcon={<AccountCircleRoundedIcon />}
                        onClick={() => signIn(MOCK_DOMAIN, username)}
                        disabled={!allowSignin}
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
                        Purchase with ViFace
                    </Button>

                    <Button
                        variant="contained"
                        startIcon={<PersonAddAlt1RoundedIcon />}
                        onClick={() => register(username)}
                        disabled={!allowSignin}
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
                        Create New Account
                    </Button>
                </Stack>
            </Box>
        </Box>
    );
};

export default LoginPage;
