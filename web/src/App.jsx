import "./index.css";
import LoginPage from "./PageLogin";
import WaitingPage from "./PageWaiting";
import SuccessPage from "./PageSuccess";
import FailedPage from "./PageFailed";
import { useState } from "react";

const STATE_LOGIN = "state_login";
const STATE_WAITING = "state_waiting";
const STATE_SUCCESS = "state_success";
const STATE_FAILED = "state_failed";

function App() {
    const [state, _] = useState(STATE_LOGIN);

    return (
        <>
            {state == STATE_LOGIN ? <LoginPage/> : <></> }
            {state == STATE_WAITING ? <WaitingPage/> : <></> }
            {state == STATE_SUCCESS ? <SuccessPage/> : <></> }
            {state == STATE_FAILED ? <FailedPage/> : <></> }
        </>
    );
}

export default App;
