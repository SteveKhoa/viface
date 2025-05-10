import "./index.css";
import LoginPage from "./PageLogin";
import WaitingPage from "./PageWaiting";
import SuccessPage from "./PageSuccess";
import FailedPage from "./PageFailed";
import { useState } from "react";

export const STATE_LOGIN = "state_login";
export const STATE_WAITING = "state_waiting";
export const STATE_SUCCESS = "state_success";
export const STATE_FAILED = "state_failed";

export const ACCESS_TOKEN_ENDPOINT = "http://localhost:8080/token";

function App() {
    const [state, setState] = useState(STATE_LOGIN);

    return (
        <>
            {state == STATE_LOGIN ? <LoginPage setState={setState} /> : <></>}
            {state == STATE_WAITING ? (
                <WaitingPage setState={setState} />
            ) : (
                <></>
            )}
            {state == STATE_SUCCESS ? (
                <SuccessPage setState={setState} />
            ) : (
                <></>
            )}
            {state == STATE_FAILED ? <FailedPage setState={setState} /> : <></>}
        </>
    );
}

export default App;
