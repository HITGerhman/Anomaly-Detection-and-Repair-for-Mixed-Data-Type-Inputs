# MEMO

Last updated: 2026-05-03 19:02:57

## 椤圭洰鎬荤洰鏍?
- 灏嗏€滄贩鍚堟暟鎹被鍨嬪紓甯告娴嬩笌淇鈥濋」鐩粠绠楁硶鍘熷瀷鎺ㄨ繘涓哄彲浜や粯銆佸彲婕旂ず銆佸彲娴嬭瘯鐨勬闈㈠簲鐢ㄣ€?- 浠?`appshell/` 浣滀负浜у搧鍖栦富璺緞锛屽舰鎴?`Python Engine + Go Backend + Wails Frontend` 鐨勭ǔ瀹氭灦鏋勩€?- 淇濈暀 `app.py` 浣滀负鏃х増 Streamlit 婕旂ず鍏ュ彛锛岀敤浜庣畻娉曢獙璇併€佺粨鏋滃鐓у拰绛旇京灞曠ず銆?- 閫愭琛ラ綈鐪熷疄鐢ㄦ埛闂幆锛氬鍏?CSV -> 璁粌/鎵弿 -> 闂绛涢€?-> 鎵归噺淇 -> 鍥炴粴 -> 鍘嗗彶鏌ョ湅銆?- 鍦ㄤ繚鐣欑幇鏈夌畻娉曡祫浜т笌宸ョ▼楠ㄦ灦鐨勫墠鎻愪笅锛岄€愭鍗囩骇涓衡€滃 agent 鍐崇瓥灞?+ 纭畾鎬у伐鍏峰眰鈥濈殑鏅鸿兘鍖栦骇鍝併€?- 鏈€缁堢洰鏍囨槸璁╃敤鎴峰敖閲忓彧闇€閫夋嫨鏂囦欢锛屽嵆鍙嚜鍔ㄨ幏寰楁壂鎻忋€佷慨澶嶃€侀獙璇併€佸洖婊氫繚鎶ゅ拰鍥捐〃鍖栬В閲婄粨鏋溿€?
## 褰撳墠閲囧彇鐨勬柟娉?
- Python 渚х粺涓€閫氳繃 JSON 鍗忚鏆撮湶 `health`銆乣train`銆乣scan_file`銆乣repair`銆乣repair_batch`銆乣rollback_repair_batch`銆?- Go 渚ц礋璐ｄ换鍔＄紪鎺掋€佸苟鍙戞帶鍒躲€佽秴鏃跺彇娑堛€丼QLite 鍘嗗彶鎸佷箙鍖栧拰鍚姩璇婃柇鑱氬悎銆?- Wails 鍓嶇鎵挎媴妗岄潰搴旂敤宸ヤ綔娴佷笌鍙鍖栦氦浜掞紝娴忚鍣ㄩ潤鎬侀瑙堟ā寮忎娇鐢?mock binding 淇濇寔鍙紑鍙戞€с€?- 鍏堜紭鍏堣ˉ宸ョ▼鍖栧熀纭€锛氬崗璁ǔ瀹氥€佸彲瑙傛祴鎬с€佸惎鍔ㄨ嚜妫€銆佺粨鏋滅洰褰?鍘嗗彶搴撶鐞嗭紝鍐嶇户缁畬鍠勫鍏ヤ綋楠屽拰淇缁撴灉绠＄悊銆?- Python 杩愯鐜鐜板湪閲囩敤鈥滃鏉句緷璧?+ 閿佸畾渚濊禆 + 鏈湴鑴氭湰鈥濅笁浠跺锛歚requirements.txt` 鐢ㄤ簬鎺㈢储锛宍requirements.lock.txt` 鐢ㄤ簬澶嶇幇锛宍scripts/setup_windows_env.ps1` 鐢ㄤ簬涓€閿垱寤?`.venv-win`銆?- 鏅鸿兘鍖栧崌绾ф柟鍚戝凡缁忔槑纭负鈥滃鐢ㄥ凡鏈夎祫浜т紭鍏堚€濓紝鍗充繚鐣?`LightGBM`銆佽鍒欐壂鎻忋€乣repair_core.py` 涓?`repair_module.py`/Gower 鐨勪环鍊硷紝閫氳繃鏂板澶?agent 缂栨帓灞傚寮轰綋楠屻€佹晥鐜囥€佸彲闈犳€т笌瑙ｉ噴鑳藉姏銆?
## 鐩墠宸插畬鎴愮殑鍏抽敭姝ラ

- 鏄庣‘鍙岃建缁撴瀯锛歚app.py` 涓烘棫鐗堟紨绀哄叆鍙ｏ紝`appshell/` 涓烘柊鐗堜骇鍝佸寲璺緞銆?- Python 寮曟搸宸插舰鎴愮ǔ瀹氳竟鐣岋紝鏀寔 `health / train / scan_file / repair / repair_batch / rollback_repair_batch`銆?- Go 鍚庣宸插叿澶囦换鍔￠槦鍒椼€佽秴鏃舵帶鍒躲€佸彇娑堛€佹渶杩戜换鍔″巻鍙插拰缁撴瀯鍖栨棩蹇楄兘鍔涖€?- Wails 鍓嶇宸叉帴鍏ョ湡瀹炵粦瀹氾紝鍏峰璁粌銆佹壂鎻忋€佹壒閲忎慨澶嶃€佸巻鍙叉煡鐪嬬瓑涓绘祦绋嬶紝骞朵繚鐣欐祻瑙堝櫒棰勮 mock銆?- 鏂板浜嗕竴濂楅潰鍚?Figma 浜や粯鐨勯潤鎬侀椤佃璁℃澘锛歚appshell/frontend/figma-home-native.html` 涓?`appshell/frontend/src/figma-home-native.css`銆?- 鍚姩鑷 v1 宸茶惤鍦帮細妗岄潰绔細鍦ㄨ繘鍏ュ洓姝ュ悜瀵煎墠妫€鏌?Python 寮曟搸銆佽繍琛屾椂渚濊禆銆佽緭鍑虹洰褰曘€丼QLite 鍜岄粯璁ゆā鍨嬬姸鎬併€?- Python 杩愯鐜涓殑 `numpy/pandas` 鎹熷潖宸蹭慨澶嶏紝`health` 鎭㈠姝ｅ父锛宍tests/python_engine` 宸叉仮澶嶅彲瀹屾暣鍥炲綊銆?- 鏂板 `tests/conftest.py`锛屼娇 `pytest tests/python_engine -q` 涓?`python -m pytest tests/python_engine -q` 閮借兘绋冲畾瀵煎叆椤圭洰鏍圭洰褰曚笅鐨?`src/` 涓?`appshell/` 妯″潡銆?- 宸插垱寤洪」鐩唴鐙珛鐜 `.venv-win`锛屽苟閫氳繃 `requirements.lock.txt` 鍥哄寲浜嗕竴濂楀湪 Windows + Python 3.11 涓嬮€氳繃鍥炲綊楠岃瘉鐨勪緷璧栭泦鍚堛€?- 鏂板 `ENVIRONMENT.md` 涓?`scripts/setup_windows_env.ps1`锛屽皢鐜鍒涘缓銆侀攣鏂囦欢瀹夎鍜屽洖褰掗獙璇佹敹鏁涗负鍙噸澶嶆墽琛岀殑浠撳簱璧勪骇銆?- 宸插舰鎴?multi-agent 闀挎湡鍗囩骇钃濆浘鐨勬牳蹇冨叡璇嗭細agent 璐熻矗鐞嗚В銆侀€夋嫨銆佽鍒掋€侀獙璇佷笌瑙ｉ噴锛岀幇鏈?Python/Go 璧勪骇缁х画鎵挎媴绋冲畾鎵ц鑱岃矗銆?- Stage 1 宸茶惤鍦版渶灏?Go 渚?`Agent Runtime`锛氬湪涓嶆柊澧?Python engine action 鐨勫墠鎻愪笅锛屾柊澧?`agent.session.plan` 涓?`agent.session.execute` 涓や釜淇濈暀缂栨帓鍔ㄤ綔锛岀敱 Go 灞傚畬鎴愪細璇濅笂涓嬫枃銆佸伐鍏疯皟搴︺€佽鍒掔敓鎴愩€侀獙璇佷笌瑙ｉ噴銆?- 宸蹭负 multi-agent 杩愯鏃舵柊澧?companion SQLite 琛?`agent_sessions` 涓?`agent_trace`锛屽湪澶嶇敤鍚屼竴涓?`APPSHELL_TASK_DB` 鐨勫悓鏃朵繚鐣欑幇鏈?`task_history` 涓昏〃涓嶅彉銆?- 宸叉柊澧?`ToolRegistry + MockPlanner + RuntimeRunner`锛屽綋鍓嶆寮忔帴鍏?`engine.scan_table` 涓?`engine.repair_batch` 涓や釜鐜版湁宸ュ叿锛岃兘澶熷熀浜庢壂鎻忕粨鏋滅敓鎴愪慨澶嶈鍒掋€佹墽琛?validation gate锛屽苟鍦ㄩ€氳繃鍚庡啀璋冪敤鐪熷疄鎵归噺淇銆?- Wails 鍚庣涓?demo CLI 宸叉柊澧?`RunAgentSession / ExecuteAgentPlan / GetAgentSession / ListAgentTrace` 鑳藉姏锛岀幇鏈?`RunTask / GetTaskStatus / CancelTask / ListTaskHistory` 涓庢棫 action 涓绘祦绋嬩繚鎸佸吋瀹广€?- Go / Python 鍥炲綊鍧囧凡瑕嗙洊 Stage 1锛歚go test ./...` 涓?`python -m pytest tests/python_engine -q` 褰撳墠鍏ㄩ儴閫氳繃銆?- Stage 2 宸插皢 `repair_module.py` / Gower 姝ｅ紡鎺ュ洖宸ュ叿灞傦細鏂板绋冲畾 action `repair_with_gower`銆丟o 渚?`engine.repair_with_gower` tool 娉ㄥ唽銆乣rule / gower / hybrid` 涓夊€欓€夎鍒掍互鍙?hybrid 澶嶅悎 rollback manifest銆?- 宸茶ˉ榻?`.gitignore` 瀵规湰鍦?Figma/娴忚鍣ㄩ獙璇佺紦瀛樼洰褰曠殑蹇界暐瑙勫垯锛岄伩鍏?`out/figma-verify/` 鐢熸垚鐗╂薄鏌?Git 鐘舵€併€?
## 褰撳墠闂

- 鍚姩鑷 warning 鎬佺幇鍦ㄤ細鏀捐杩涘叆搴旂敤锛屼絾涓荤晫闈㈤噷杩樻病鏈夆€滈噸鏂版煡鐪嬪惎鍔ㄨ瘖鏂€濈殑鎸佷箙鍏ュ彛銆?- Windows 鎵撳寘閾捐矾浠嶆湭瀹屽叏鏀跺彛锛宍appshell/build/windows/` 杩橀渶瑕佸畬鎴愮湡姝ｅ彲鍒嗗彂鐨勬瀯寤轰笌瀹夎楠岃瘉銆?- 鐙珛鐜宸茬粡鑳戒繚璇佸紑鍙戜笌娴嬭瘯鍦?Windows + Python 3.11 涓嬬殑涓€鑷存€э紝浣嗏€滃垎鍙戝埌涓嶅悓鏈哄櫒鍚庝篃缁濆涓€鑷粹€濅粛鐒堕渶瑕佸畨瑁呭寘闃舵鎶?Python runtime 鍜岄獙璇佽繃鐨勪緷璧栦竴璧锋墦杩涘幓锛屼笉鑳藉彧渚濊禆鐩爣鏈哄櫒鏈湴 Python銆?- 浠撳簱閲岃繕淇濈暀鐫€涓€涓棫鐨?Linux 椋庢牸 `.venv-appshell`锛屽綋鍓嶆病鏈夎閲囩敤锛屽悗缁彲浠ヨ鎯呭喌娓呯悊鎴栧湪鏂囨。涓槑纭爣娉ㄤ负閬楃暀璧勪骇銆?- multi-agent 鐨?Stage 2 铏藉凡瀹屾垚 Gower 宸ュ叿鍖栥€佸弻璺瘮杈冧笌 hybrid 鎵ц锛屼絾鐪熷疄 LLM provider銆侀暱鏈?memory銆佸墠绔竴閿棴鐜帴鍏ヤ笌鍥捐〃鍖栬В閲婁粛鏈紑濮嬨€?- 鈥滅敤鎴峰彧闇€閫夋枃浠堕潤鍊欑粨鏋溾€濈殑闂幆浣撻獙杩樻病鏈夋寮忔帴鍏ュ綋鍓?Wails 涓绘祦绋嬶紝agent 鍏ュ彛鐩墠浠嶅仠鐣欏湪鍚庣 API / demo CLI 灞傘€?- 鍥捐〃鍖栬В閲婁綋绯讳粛鏈繘鍏ユ寮忓疄鐜帮紝寮傚父鐞嗚В鍥俱€佷慨澶嶆敹鐩婂浘涓?agent 杞ㄨ抗鍥惧皻鏈帴鍒版闈㈢椤甸潰銆?
## Update 2026-03-16 19:58:35

- 鏀瑰姩鏃ユ湡锛?026-03-16 19:58:35
- 鏀瑰姩鍐呭绠€杩帮細瀹屾垚 `MULTI_AGENT_BLUEPRINT.md` 涓?Stage 4鈥滅粺涓€瑙ｉ噴绯荤粺涓庡浘琛ㄧ郴缁燂紙Wails 浼樺厛锛夆€濈殑绗竴杞寮忚惤鍦帮紝鍦ㄤ笉鏂板 Python stable action銆佷笉涓柇褰撳墠涓讳慨澶嶉摼璺殑鍓嶆彁涓嬶紝寮曞叆缁熶竴 `presentation` 琛ㄨ揪灞傘€佺郴缁熺骇鍥捐〃鐩綍銆乄ails 浼樺厛缁撴灉娓叉煋浠ュ強 Streamlit 瀵?`presentation.json` 鐨勮交閲忓鐢ㄥ叆鍙ｃ€?- 鐩稿叧妯″潡/鏂囦欢锛?  - `appshell/backend/internal/presentation/types.go`
  - `appshell/backend/internal/presentation/helpers.go`
  - `appshell/backend/internal/presentation/builder_common.go`
  - `appshell/backend/internal/presentation/builder_scan_repair.go`
  - `appshell/backend/internal/presentation/builder_agent.go`
  - `appshell/backend/internal/presentation/artifact.go`
  - `appshell/backend/internal/presentation/builder_test.go`
  - `appshell/backend/cmd/wails/app.go`
  - `appshell/backend/cmd/wails/app_test.go`
  - `appshell/backend/internal/task/service.go`
  - `appshell/frontend/index.html`
  - `appshell/frontend/src/main.js`
  - `appshell/frontend/src/style.css`
  - `app.py`
  - `PRESENTATION_CATALOG.md`
  - `MEMO.md`
- 宸茶В鍐崇殑闂/鏂板鍔熻兘锛?  - 鏂板 Go 渚?`internal/presentation` 鍖咃紝姝ｅ紡瀹氫箟 `PresentationBundle / Highlight / Section / ChartSpec`锛屽苟鎶?`scan_file / repair_batch / repair_with_gower / agent.session.*` 鐨勫師濮嬬粨鏋滅粺涓€杞崲涓?`presentation` 琛ㄨ揪灞傘€?  - `task.Service` 鐜板湪浼氬湪浠诲姟瀹屾垚鍚庤嚜鍔ㄥ悜 `response.result` 闄勫姞 `presentation`锛屽苟鍦ㄦ湁杈撳嚭鐩綍鎴栬緭鍑?CSV 鏃堕澶栧啓鍑?`presentation.json`锛屼綔涓哄伐绋嬬涓庣瓟杈╃鍏变韩璧勪骇銆?  - `GetAgentSession` 鐜板凡鍚屾杩斿洖 `presentation` 涓庡彲閫夌殑 `presentation_artifact`锛岃 session 绾х粨鏋滃拰鏅€?task 缁撴灉鍏变韩鍚屼竴濂楄〃杈炬ā鍨嬨€?  - 鏂板鏍圭洰褰?`PRESENTATION_CATALOG.md`锛屽喕缁?Stage 4 鐨勮В閲婃ā鏉?ID銆佸浘琛?ID銆佽緭鍏ュ瓧娈点€佹樉绀烘潯浠朵笌榛樿鏂囨鍙ｅ緞銆?  - Wails 鍓嶇缁撴灉椤靛凡鎺ュ叆缁熶竴 `renderPresentationBundle(...)`锛屾壂鎻忕粨鏋溿€佹櫘閫氫慨澶嶇粨鏋滀互鍙?agent 鑷姩闂幆缁撴灉閮借兘浼樺厛鏄剧ず鈥滄憳瑕?+ 鎸囨爣鍗?+ 鍥捐〃鍗?+ 瑙ｉ噴娈佃惤鈥濓紝鍘熷 JSON 鍜屾槑缁嗚〃缁х画淇濈暀涓洪珮绾х粏鑺傘€?  - 鍓嶇琛ラ綈浜?`repair_with_gower`銆乣agent.session.plan`銆乣agent.session.execute`銆乣agent.session.auto` 鐨勬剰鍥炬槧灏勶紝浣垮巻鍙蹭换鍔″拰 agent 缁撴灉涔熻兘钀藉埌缁熶竴缁撴灉瑙嗗浘銆?  - `app.py` 鏂板杞婚噺 `Presentation Viewer` 椤甸潰锛屽彲鐩存帴璇诲彇 `presentation.json` 骞跺睍绀烘憳瑕併€佽В閲婃钀藉拰绠€鍖栧浘琛紱鍘熸湁 `ROC / Confusion Matrix / Feature Importance / SHAP` 椤甸潰淇濇寔涓嶅彉銆?  - 鍥炲綊缁撴灉锛歚go test ./...` 閫氳繃锛沗node --check appshell/frontend/src/main.js` 閫氳繃锛沗python -m py_compile app.py` 閫氳繃锛沗python -m pytest tests/python_engine -q` 閫氳繃锛屽綋鍓嶄负 `29 passed`銆?- 寰呭鐞嗕簨椤癸細
  - 灏?Stage 4 鏂板鐨?`presentation` 缁撴灉姝ｅ紡鎺ュ叆褰撳墠 Wails 榛樿涓绘祦绋嬪叆鍙ｏ紝杩涗竴姝ラ潬杩戔€滃彧閫夋枃浠跺悗闈欏€欑粨鏋溾€濈殑浜у搧浣撻獙銆?  - 涓?Wails 缁撴灉椤佃ˉ鏇寸粏鐨勬墜鍔ㄩ獙鏀朵笌瑙嗚鎵撶（锛屽挨鍏舵槸 heatmap銆乼imeline 鍜?rollback 璺緞涓嬬殑鎺掔増缁嗚妭銆?  - 鍚庣画缁х画鎺ㄨ繘鐪熷疄 provider-agnostic planner銆侀暱鏈?memory 涓庡鎵圭瓥鐣ワ紝浣嗕繚鎸?Stage 3 宸插舰鎴愮殑 validation-first / rollback-first 瀹夊叏杈圭晫涓嶉€€鍖栥€?
## Update 2026-03-16 18:56:06

- 鏀瑰姩鏃ユ湡锛?026-03-16 18:56:06
- 鏀瑰姩鍐呭绠€杩帮細瀹屾垚 `MULTI_AGENT_BLUEPRINT.md` 涓?Stage 3鈥滃舰鎴愰獙璇佷紭鍏堢殑鍏ㄨ嚜鍔ㄩ棴鐜€濈殑绗竴杞寮忚惤鍦帮紝鍦ㄤ繚鐣?Stage 1/2 鏃㈡湁 `agent.session.plan / agent.session.execute` 閾捐矾涓嶅彉鐨勫墠鎻愪笅锛屾柊澧?`agent.session.auto` 鑷姩闂幆鍏ュ彛锛屽苟鎶娾€滈鏍￠獙 -> 鎵ц -> 澶嶆壂 -> 鍚庨獙楠岃瘉 -> 鑷姩鍥炴粴 -> 瀹¤鐣欑棔鈥濆畬鏁存帴鍏?Go 渚?Agent Runtime銆?- 鐩稿叧妯″潡/鏂囦欢锛?  - `appshell/backend/internal/agent/actions.go`
  - `appshell/backend/internal/agent/runtime_runner.go`
  - `appshell/backend/internal/agent/planning_flow.go`
  - `appshell/backend/internal/agent/auto_helpers.go`
  - `appshell/backend/internal/agent/auto_session.go`
  - `appshell/backend/internal/agent/runtime_runner_test.go`
  - `appshell/backend/cmd/wails/app.go`
  - `appshell/backend/cmd/wails/app_test.go`
  - `appshell/backend/cmd/demo/main.go`
  - `appshell/backend/internal/task/service.go`
  - `appshell/backend/internal/task/service_test.go`
  - `MEMO.md`
- 宸茶В鍐崇殑闂/鏂板鍔熻兘锛?  - 鏂板淇濈暀鍔ㄤ綔 `agent.session.auto`锛岀敱 Go 渚?`RuntimeRunner` 鎷︽埅锛屽舰鎴愮湡姝ｇ殑鑷姩闂幆鎵ц闈紝鑰屼笉鏂板浠讳綍 Python stable action銆?  - 鏂板 `RunAgentAutofixSession` Wails 缁戝畾鏂规硶锛屽苟璁?demo CLI 鏀寔 `-action agent.session.auto`锛屼究浜庡湪涓嶆敼鐜版湁鍓嶇涓绘祦绋嬬殑鎯呭喌涓嬬嫭绔嬮獙璇佽嚜鍔ㄦā寮忋€?  - 灏?planning 閫昏緫鎶藉彇涓哄叡浜?`runPlanningFlow()`锛岃 `agent.session.plan` 涓?`agent.session.auto` 澶嶇敤鍚屼竴濂?scan / rule preview / gower preview / multi-candidate planning 娴佺▼銆?  - 鑷姩妯″紡鐜板凡鍥哄畾鎵ц `preview validation -> execute -> post scan -> post validation`锛屽苟鍦ㄤ笉婊¤冻鎺ュ彈鏉′欢鏃惰嚜鍔ㄨ皟鐢?`rollback_repair_batch`锛岄粯璁ゅ彧鎭㈠杈撳嚭浜х墿锛屼笉瑕嗙洊鍘熷杈撳叆 CSV銆?  - 涓鸿嚜鍔ㄥ洖婊氳ˉ榻?rejected output snapshot 閫昏緫锛屽け璐ヤ骇鐗╀細鍏堝鍒跺埌 rollback 鐩綍涓嬬殑 `.rejected.csv`锛屽啀鎵ц鎭㈠锛屼究浜庡悗缁璁′笌澶嶇洏銆?  - `agent_sessions` 涓婁笅鏂囩幇宸叉矇娣€ `baseline_scan / preview_validation / post_scan / post_validation / rollback_summary / final_verdict / rejected_output_snapshot` 绛?Stage 3 鍏抽敭瀛楁銆?  - `agent_trace` 鏂板 `rollback_decision / rollback_executed` 杞ㄨ抗绫诲瀷锛宍validation` 浜嬩欢涔熻ˉ鍏呬簡 `phase=preview|post_execute`锛岃鑷姩闂幆鍏峰鍙噸鏀俱€佸彲瀹¤鍩虹銆?  - `task.Service` 鏂板 `agent_rescan / agent_post_validate / agent_rollback` 闃舵鍚嶆槧灏勶紝鏃ц繘搴︿綋绯绘棤闇€鏀圭増鍗冲彲灞曠ず Stage 3 鏂伴樁娈点€?  - 鏂板骞堕€氳繃浜?Stage 3 鍏抽敭鍥炲綊锛氳嚜鍔ㄩ棴鐜垚鍔熴€乸review 鎷掔粷銆乸ost-validation 澶辫触鍚庤嚜鍔ㄥ洖婊氥€乺ollback 澶辫触鍥涙潯鏍稿績璺緞銆?  - 鍥炲綊缁撴灉锛歚go test ./...` 閫氳繃锛沗python -m pytest tests/python_engine -q` 閫氳繃锛屽綋鍓嶄负 `29 passed`銆?- 寰呭鐞嗕簨椤癸細
  - 灏?`RunAgentAutofixSession` 姝ｅ紡鎺ュ叆 Wails 涓诲悜瀵兼垨榛樿鍏ュ彛锛屽悜鈥滅敤鎴峰彧閫夋枃浠跺悗闈欏€欑粨鏋溾€濈殑涓讳綋楠岀户缁帹杩涖€?  - 鍦ㄨ嚜鍔ㄩ棴鐜箣涓婄户缁ˉ榻愭洿涓板瘜鐨勫浘琛ㄥ寲瑙ｉ噴锛屽寘鎷紓甯哥悊瑙ｅ浘銆佷慨澶嶆敹鐩婂浘銆侀獙璇佺粨鏋滃浘鍜?agent trace 鏃堕棿绾裤€?  - 缁х画鎺ㄨ繘鐪熷疄 provider-agnostic LLM planner銆侀暱鏈?memory 涓庡鎵圭瓥鐣ワ紝浣嗕繚鎸?Stage 3 褰㈡垚鐨?validation-first / rollback-first 瀹夊叏杈圭晫涓嶉€€鍖栥€?
## Update 2026-03-16 17:42:59

- 鏀瑰姩鏃ユ湡锛?026-03-16 17:42:59
- 鏀瑰姩鍐呭绠€杩帮細瀹屾垚 `MULTI_AGENT_BLUEPRINT.md` 涓?Stage 2鈥滃皢 Gower 鑳藉姏閲嶆柊寮曞叆涓烘寮忓伐鍏峰苟鎺ュ叆鍙岃矾姣旇緝鈥濈殑绗竴杞寮忚惤鍦帮紝鍦ㄤ繚鐣欑幇鏈?`RunTask -> task.Service -> engine.Runner -> Python engine action` 涓婚摼璺殑鍓嶆彁涓嬶紝鏂板绋冲畾 action `repair_with_gower`锛屾妸 `repair_module.py` 閲嶆瀯涓哄彲澶嶇敤鐨?Gower 閭诲眳寤鸿妯″潡锛屽苟鎶?Go 渚?agent plan 鍗囩骇涓?`rule / gower / hybrid` 涓夊€欓€夋瘮杈冧笌 hybrid 鎵ц銆?- 鐩稿叧妯″潡/鏂囦欢锛?  - `src/repair_module.py`
  - `appshell/core/python_engine/action_catalog.py`
  - `appshell/core/python_engine/engine_core.py`
  - `tests/python_engine/test_action_catalog.py`
  - `tests/python_engine/test_engine_cli.py`
  - `appshell/backend/internal/engine/actions.go`
  - `appshell/backend/internal/engine/actions_test.go`
  - `appshell/backend/internal/agent/types.go`
  - `appshell/backend/internal/agent/helpers.go`
  - `appshell/backend/internal/agent/planner.go`
  - `appshell/backend/internal/agent/mock_planner.go`
  - `appshell/backend/internal/agent/mock_planner_test.go`
  - `appshell/backend/internal/agent/tool_registry.go`
  - `appshell/backend/internal/agent/tool_registry_test.go`
  - `appshell/backend/internal/agent/runtime_runner.go`
  - `appshell/backend/internal/agent/runtime_runner_test.go`
  - `appshell/backend/internal/task/service.go`
  - `appshell/backend/cmd/wails/app.go`
  - `appshell/backend/cmd/wails/app_test.go`
  - `appshell/backend/cmd/demo/main.go`
  - `MEMO.md`
- 宸茶В鍐崇殑闂/鏂板鍔熻兘锛?  - 鏂板绋冲畾 Python engine action `repair_with_gower`锛屽苟灏嗙ǔ瀹?action 椤哄簭鏇存柊涓?`health / train / repair / scan_file / repair_batch / repair_with_gower / rollback_repair_batch`銆?  - 灏?`src/repair_module.py` 鏁寸悊涓哄彲琚?`engine_core.py` 璋冪敤鐨?Gower 閭诲眳寤鸿妯″潡锛屾敮鎸?`uniform / model_importance / custom` 涓夌鏉冮噸妯″紡锛屽苟鍦?`model_dir` 鍙敤鏃朵娇鐢?LightGBM feature importance 鍔犳潈銆?  - `repair_with_gower` 鐜板凡鏀寔 `missing_values / numeric_outlier / rare_category` 涓夌被闂锛岃繑鍥炰笌 `repair_batch` 鍚屽彛寰勭殑 `comparison`锛屽苟鏂板 `neighbor_evidence`銆乣gower_strategy` 涓?versioned rollback manifest銆?  - `rollback_repair_batch` 鐜板凡鍏煎鏃?rule manifest 浠ュ強鏂扮殑 Gower / hybrid v2 manifest锛屽苟鍦ㄨ繑鍥炵粨鏋滀腑鏆撮湶 `manifest_version`銆乣source_tool_id` 涓?`issue_source_map`銆?  - Go 渚?`ToolRegistry` 宸叉敞鍐?`engine.repair_with_gower`锛宍KnownActions()`銆乄ails `normalizeRequest()` 涓?demo CLI 鐜板凡鏀寔鐩存帴鎻愪氦 `repair_with_gower`銆?  - Go 渚?`MockPlanner` 涓?`RuntimeRunner` 宸蹭粠鍗曞€欓€夎鍒掑崌绾т负 `rule / gower / hybrid` 涓夊€欓€夋瘮杈冿紝鏀寔 `agent_retrieve`銆乣agent_compare` 涓や釜鏂伴樁娈碉紝骞惰兘鎸?issue 绾ф潵婧愭墽琛?hybrid 淇涓庣敓鎴愬鍚?rollback manifest銆?  - 宸插畬鎴愬洖褰掗獙璇侊細`go test ./...` 閫氳繃锛沗python -m pytest tests/python_engine -q` 閫氳繃锛屽綋鍓嶄负 `29 passed`銆?- 寰呭鐞嗕簨椤癸細
  - 涓?`repair_with_gower` 澧炲姞鏇寸粏绮掑害鐨勫紓甯哥被鍨嬭鐩栦笌鏇寸ǔ瀹氱殑閭诲眳绛涢€夌瓥鐣ワ紝閫愭鎵╁睍鍒?Stage 2 涔嬪鐨勫鏉傞棶棰樼被鍨嬨€?  - 鍦ㄥ墠绔寮忔帴鍏?`RunAgentSession / ExecuteAgentPlan / GetAgentSession / ListAgentTrace`锛屾妸涓夊€欓€夋瘮杈冦€乿alidation gate 涓?hybrid 鎵ц缁撴灉杞垚鍙鍖栦氦浜掋€?  - 缁х画鎺ㄨ繘鐪熷疄 provider-agnostic LLM planner銆侀暱鏈?memory銆佸鎵圭瓥鐣ヤ笌缁撴灉鍥捐〃浣撶郴锛岃鈥滅敤鎴峰彧閫夋枃浠堕潤鍊欑粨鏋溾€濈殑闂幆浣撻獙鐪熸钀藉埌 Wails 涓绘祦绋嬨€?
## Update 2026-03-16 15:13:53

- 鏀瑰姩鏃ユ湡锛?026-03-16 15:13:53
- 鏀瑰姩鍐呭绠€杩帮細瀹屾垚 `MULTI_AGENT_BLUEPRINT.md` 涓?Stage 1鈥滃湪鐜版湁宸ュ叿澶栧鍔?Agent Runtime鈥濈殑绗竴杞寮忚惤鍦帮紝鍦ㄤ笉鐮村潖鐜版湁 `RunTask -> task.Service -> engine.Runner -> Python engine action` 閾捐矾鐨勫墠鎻愪笅锛屼负 Go 鍚庣琛ラ綈 agent runtime銆佷細璇?杞ㄨ抗鎸佷箙鍖栥€乿alidation gate銆佹柊澧?Wails 缁戝畾鏂规硶涓?demo CLI 楠岃瘉鍏ュ彛銆?- 鐩稿叧妯″潡/鏂囦欢锛?  - `appshell/backend/internal/agent/actions.go`
  - `appshell/backend/internal/agent/types.go`
  - `appshell/backend/internal/agent/store.go`
  - `appshell/backend/internal/agent/sqlite_store.go`
  - `appshell/backend/internal/agent/tool_registry.go`
  - `appshell/backend/internal/agent/planner.go`
  - `appshell/backend/internal/agent/mock_planner.go`
  - `appshell/backend/internal/agent/runtime_runner.go`
  - `appshell/backend/internal/agent/tool_registry_test.go`
  - `appshell/backend/internal/agent/mock_planner_test.go`
  - `appshell/backend/internal/agent/sqlite_store_test.go`
  - `appshell/backend/internal/agent/runtime_runner_test.go`
  - `appshell/backend/cmd/wails/app.go`
  - `appshell/backend/cmd/wails/app_test.go`
  - `appshell/backend/cmd/wails/startup_checks.go`
  - `appshell/backend/cmd/demo/main.go`
  - `appshell/backend/internal/task/service.go`
  - `appshell/backend/internal/task/service_test.go`
  - `MEMO.md`
- 宸茶В鍐崇殑闂/鏂板鍔熻兘锛?  - 鏂板 Go 渚?`internal/agent` 鍖咃紝姝ｅ紡钀藉湴 `RuntimeRunner / ToolRegistry / Planner / SQLiteStore` 杩欑粍 Stage 1 鏍稿績瀵硅薄銆?  - 鍦ㄤ笉鏀?Python engine action 闆嗗悎銆佷笉鏀?`health` 杩斿洖鍒楄〃鐨勫墠鎻愪笅锛屾柊澧?`agent.session.plan` 涓?`agent.session.execute` 涓や釜 Go 缂栨帓鍔ㄤ綔锛屽苟淇濇寔鏅€?action 浠嶇洿鎺ラ€忎紶鍒板簳灞?`engine.Runner`銆?  - 鏂板 `agent_sessions` 涓?`agent_trace` companion tables锛屽紑濮嬪湪涓?`task_history` 鍚屼竴涓?SQLite 鏂囦欢涓矇娣€ session 涓婁笅鏂囥€乤gent 鍐崇瓥銆乼ool call銆乿alidation 涓?completion/failure 杞ㄨ抗銆?  - 鏂板 `MockPlanner`锛屽綋鍓嶄細鍩轰簬 `scan_file` 缁撴灉鍙寫閫?`missing_values / numeric_outlier / rare_category` 涓夌被 `repair_batch` 宸叉敮鎸侀棶棰橈紝杈撳嚭鏍囧噯鍖?`AgentPlan` 涓庤В閲婃枃鏈€?  - 鏂板 validation gate锛歚agent.session.execute` 浼氬厛鎵ц `repair_batch(plan_only=true)` 棰勯獙璇侊紝鍙湁 `resolved_issue_count > 0` 涓?`after_issue_count <= before_issue_count` 鏃舵墠杩涘叆鐪熷疄鎵ц锛屽苟鍥哄畾寮€鍚?`enable_rollback=true`銆?  - Wails 鍚庣鏂板 `RunAgentSession`銆乣ExecuteAgentPlan`銆乣GetAgentSession`銆乣ListAgentTrace` 鍥涗釜缁戝畾鏂规硶锛沝emo CLI 鏂板 `-action agent.session.plan` / `-action agent.session.execute` 浠ュ強 `-goal / -session-id / -plan-id / -csv / -output` 鍙傛暟鏀寔銆?  - `task.Service` 鏂板 `agent_intent / agent_profile / agent_scan / agent_strategy / agent_plan / agent_validate / agent_execute / agent_explain` 闃舵鏄犲皠锛屾棫鐨勪换鍔¤繘搴︿綋绯诲彲浠ョ洿鎺ユ壙杞?agent synthetic progress銆?  - 宸插畬鎴愬洖褰掗獙璇侊細`go test ./...` 閫氳繃锛沗python -m pytest tests/python_engine -q` 閫氳繃锛屽綋鍓嶄负 `26 passed`銆?- 寰呭鐞嗕簨椤癸細
  - 鍦?Stage 2 涓皢 `repair_module.py` / Gower 閲嶆柊浠?`repair_with_gower` 閫傞厤鍣ㄥ舰寮忔帴鍏?Tool Layer锛岃 Retrieval Agent 鑳戒娇鐢ㄨ繎閭昏瘉鎹€屼笉鏄彧渚濊禆瑙勫垯淇銆?  - 涓?`Agent Runtime` 棰勭暀鐪熷疄 provider-agnostic LLM planner 鎺ュ彛瀹炵幇锛岄€愭鏇挎崲褰撳墠妯℃澘鍖?`MockPlanner` 涓庤В閲婅緭鍑恒€?  - 灏?`RunAgentSession / ExecuteAgentPlan / GetAgentSession / ListAgentTrace` 鐪熸鎺ュ叆 Wails 鍓嶇涓绘祦绋嬶紝閫愭鎻愪緵鈥滀笂浼犳枃浠跺悗闈欏€欎匠闊斥€濈殑榛樿浜や簰妯″紡銆?  - 璁捐骞跺疄鐜板浘琛ㄥ寲瑙ｉ噴瑙嗗浘锛屼紭鍏堣鐩栧紓甯稿垎甯冦€佷慨澶嶅墠鍚庡姣斻€乿alidation 缁撴灉鍜?agent trace 鏃堕棿绾裤€?
## Update 2026-03-16 14:18:17

- 鏀瑰姩鏃ユ湡锛?026-03-16 14:18:17
- 鏀瑰姩鍐呭绠€杩帮細瀹屾垚 `MULTI_AGENT_BLUEPRINT.md` 涓?Stage 0鈥滀繚鐣欎笌鍖呰鐜版湁璧勪骇鈥濈殑绗竴杞寮忚惤鍦帮紝鍦ㄤ笉鏀瑰彉澶栭儴鍗忚鍜屽墠绔彲瑙佹祦绋嬬殑鍓嶆彁涓嬶紝寤虹珛 Python action 鍏冩暟鎹洰褰曘€丟o action 甯搁噺杈圭晫銆丼tage 0 宸ュ叿鍩虹鏂囨。涓庡搴斿洖褰掓祴璇曘€?- 鐩稿叧妯″潡/鏂囦欢锛?  - `appshell/core/python_engine/action_catalog.py`
  - `appshell/core/python_engine/engine_service.py`
  - `appshell/core/python_engine/engine_core.py`
  - `appshell/backend/internal/engine/actions.go`
  - `appshell/backend/internal/engine/actions_test.go`
  - `appshell/backend/cmd/wails/app.go`
  - `appshell/backend/cmd/wails/app_test.go`
  - `appshell/backend/cmd/demo/main.go`
  - `tests/python_engine/test_action_catalog.py`
  - `TOOL_LAYER_FOUNDATION.md`
  - `README.md`
  - `appshell/README.md`
  - `MULTI_AGENT_BLUEPRINT.md`
  - `MEMO.md`
- 宸茶В鍐崇殑闂/鏂板鍔熻兘锛?  - 鏂板 `ActionSpec` 鐩綍锛屾寮忓喕缁?`health / train / repair / scan_file / repair_batch / rollback_repair_batch` 鍏釜绋冲畾 action锛屽苟涓烘瘡涓?action 寤虹珛鏈潵 canonical tool id銆佽緭鍏ュ瓧娈点€佷骇鐗╀笌绠楁硶璧勪骇鏄犲皠銆?  - `engine_service.supported_actions()`銆乤ction 璺敱娉ㄥ唽浠ュ強 `action_health` 鐨勫叕寮€ action 鍒楄〃鏀逛负浠庣粺涓€鐩綍鐢熸垚锛屾秷闄ょ‖缂栫爜婕傜Щ椋庨櫓锛屽悓鏃朵繚鎸?`health` 杩斿洖椤哄簭浠嶄负 `health, train, repair, scan_file, repair_batch, rollback_repair_batch`銆?  - Go 渚ф柊澧?`ActionName` 涓?`KnownActions()`锛屽苟灏?Wails / demo 涓や釜鍏ュ彛鏀逛负寮曠敤缁熶竴鍔ㄤ綔甯搁噺锛屼负 Stage 1 鐨?Tool Layer 涓?Agent Runtime 閾鸿矾銆?  - 鏂板 `TOOL_LAYER_FOUNDATION.md`锛屾妸 action銆乼ool銆乤lgorithm asset銆乤rtifact 鍥涘姒傚康鍜屽綋鍓嶈祫浜ф槧灏勬寮忔枃妗ｅ寲銆?  - 鏇存柊鏍圭洰褰曚笌 `appshell/` 璇存槑鏂囨。锛屾槑纭?Stage 0 鍙仛淇濈暀涓庡寘瑁咃紝涓嶅紩鍏ョ敤鎴峰彲瑙佹櫤鑳藉寲鍏ュ彛鍙樺寲銆?  - 鏂板 Python/Go 鍥炲綊娴嬭瘯锛岃鐩?action catalog銆佺ǔ瀹?action 闆嗗悎鍜屽叆鍙ｅ父閲忓紩鐢ㄣ€?- 寰呭鐞嗕簨椤癸細
  - 杩愯骞剁‘璁?Stage 0 鐨?Python 涓?Go 鍥炲綊娴嬭瘯鍏ㄩ儴閫氳繃锛岄獙璇佹湰杞暣鐞嗘湭寮曞叆琛屼负鍥為€€銆?  - 鍦?Stage 1 涓熀浜庡綋鍓?`ActionSpec` 涓?Go action 甯搁噺缁х画璁捐 `Agent Runtime / Tool Registry / Task Trace / Validation Gate` 鐨勬渶灏忓疄鐜般€?  - 鍦ㄤ笉鏂板瀵瑰 action 鐨勫墠鎻愪笅锛屽悗缁皢 `repair_module.py` 涓殑 Gower 鑳藉姏鍖呰涓?`repair_with_gower` 閫傞厤鍣ㄥ瀷宸ュ叿銆?
## Update 2026-03-16 13:16:19

- 鏀瑰姩鏃ユ湡锛?026-03-16 13:16:19
- 鏀瑰姩鍐呭绠€杩帮細淇ˉ `.gitignore` 涓仐婕忕殑鏈湴楠岃瘉鐢熸垚鐗╁拷鐣ヨ鍒欙紝瑙ｅ喅 `out/figma-verify/` 澶ч噺娴忚鍣ㄧ紦瀛樻枃浠舵薄鏌?Git 鐘舵€佺殑闂銆?- 鐩稿叧妯″潡/鏂囦欢锛?  - `.gitignore`
  - `MEMO.md`
- 宸茶В鍐崇殑闂/鏂板鍔熻兘锛?  - 鏂板瀵?`out/figma-verify/` 鐨勫拷鐣ヨ鍒欙紝閬垮厤鏈湴 Figma/娴忚鍣ㄩ獙璇佽繃绋嬩腑浜х敓鐨勫ぇ閲?profile銆乧ache銆乧rashpad 鍜?leveldb 鏂囦欢杩涘叆 Git 鐘舵€併€?  - 淇濇寔 `out/` 鐩綍鍏朵綑鍐呭涓嶈涓€鍒€鍒囧拷鐣ワ紝閬垮厤璇激鏈潵鍙兘闇€瑕佷繚鐣欑殑浜х墿銆?  - 灏嗘湰娆?Git 宸ヤ綔鍖烘不鐞嗗悓姝ヨ褰曞埌椤圭洰绾锛屼究浜庡悗缁拷韪拷鐣ョ瓥鐣ュ彉鏇淬€?- 寰呭鐞嗕簨椤癸細
  - 濡傚悗缁?`out/` 涓嬪嚭鐜版洿澶氬悓绫讳复鏃堕獙璇佺洰褰曪紝鍐嶈瘎浼版槸鍚︽娊璞′负鏇撮€氱敤鐨勫拷鐣ヨ鍒欍€?  - 缁х画鎺ㄨ繘 multi-agent 杩愯鏃躲€佸伐鍏锋敞鍐屽眰涓庡浘琛ㄥ寲缁撴灉瑙嗗浘鐨勬寮忓疄鐜般€?
## Update 2026-03-16 11:32:58

- 鏀瑰姩鏃ユ湡锛?026-03-16 11:32:58
- 鏀瑰姩鍐呭绠€杩帮細鏂板闈㈠悜闀挎湡鏅鸿兘鍖栧崌绾х殑椤圭洰绾查鏂囦欢锛屾槑纭€滃鐢ㄧ幇鏈夌畻娉曚笌宸ョ▼璧勪骇銆侀€氳繃澶?agent 鎻愬崌浣撻獙涓庡彲闈犳€с€佹渶缁堝疄鐜扮敤鎴烽€夋枃浠跺悗鑷姩闂幆澶勭悊鈥濈殑鎬讳綋鏂瑰悜銆?- 鐩稿叧妯″潡/鏂囦欢锛?  - `MULTI_AGENT_BLUEPRINT.md`
  - `MEMO.md`
- 宸茶В鍐崇殑闂/鏂板鍔熻兘锛?  - 鏂板鏍圭洰褰?`MULTI_AGENT_BLUEPRINT.md`锛屼綔涓哄悗缁?multi-agent 鍗囩骇鏀归€犵殑鏈€楂樺眰绾查鏂囦欢銆?  - 鏄庣‘浜嗘櫤鑳藉寲鍗囩骇鐨勭洰鏍囦笉鏄浛鎹㈢幇鏈夌畻娉曪紝鑰屾槸鍦ㄧ幇鏈?`LightGBM`銆佽鍒欐壂鎻忋€乣repair_core.py` 涓?`repair_module.py`/Gower 鍩虹涓婃柊澧炲喅绛栧眰涓庤В閲婂眰銆?  - 鏄庣‘浜嗘湭鏉ョ洰鏍囦綋楠岋細鐢ㄦ埛灏介噺鍙渶閫夋嫨鏂囦欢锛岀郴缁熻嚜鍔ㄥ畬鎴愭壂鎻忋€佷慨澶嶈鍒掋€佹墽琛屻€侀獙璇併€佸洖婊氫繚鎶ゅ拰鏈€缁堣В閲娿€?  - 鏄庣‘浜嗘湭鏉ュ浘琛ㄤ綋绯绘柟鍚戯細缁ф壙 Streamlit 涓?Wails 鐜版湁鍙鍖栬祫浜э紝骞舵墿灞曞紓甯哥悊瑙ｃ€佷慨澶嶆敹鐩婁笌鎵ц杞ㄨ抗鍥捐〃銆?- 寰呭鐞嗕簨椤癸細
  - 灏?`MULTI_AGENT_BLUEPRINT.md` 涓殑闀挎湡钃濆浘閫愭鎷嗚В涓哄彲瀹炴柦鐨勯樁娈佃鍒掍笌鎺ュ彛璁捐銆?  - 璁捐 Go 渚?`Agent Runtime / Tool Registry / Task Trace / Validation Gate` 鐨勬渶灏忓彲鐢ㄥ疄鐜版柟妗堛€?  - 灏?`repair_module.py` 涓殑 Gower 閭诲眳妫€绱㈣兘鍔涢噸鏂版帴鍏ユ寮忓伐鍏峰眰銆?  - 鍦ㄥ墠绔腑閫愭寮曞叆鈥滀竴閿笂浼犮€侀潤鍊欑粨鏋溾€濈殑鍩虹妯″紡涓庢洿涓板瘜鐨勫浘琛ㄧ粨鏋滆鍥俱€?
## Update 2026-03-12 14:54:25

- 鏀瑰姩鏃ユ湡锛?026-03-12 14:54:25
- 鏀瑰姩鍐呭绠€杩帮細灏?Python 杩愯鐜鏀舵暃涓洪」鐩唴鐙珛鐜鏂规锛岃ˉ榻愰攣鏂囦欢銆佺幆澧冭剼鏈拰鐜璇存槑鏂囨。锛屽苟鍦ㄩ殧绂荤幆澧冧腑瀹屾垚鐪熷疄鍥炲綊楠岃瘉銆?- 鐩稿叧妯″潡/鏂囦欢锛?  - `requirements.lock.txt`
  - `ENVIRONMENT.md`
  - `scripts/setup_windows_env.ps1`
  - `README.md`
  - `appshell/README.md`
  - `MEMO.md`
- 宸茶В鍐崇殑闂/鏂板鍔熻兘锛?  - 鏂板缓 `.venv-win` 椤圭洰鏈湴鐜锛屽苟鍦ㄥ叾涓畨瑁呬竴濂楀畬鏁翠緷璧栵紝纭繚涓嶅啀渚濊禆 Anaconda `base` 杩愯椤圭洰銆?  - 鍦?`.venv-win` 涓獙璇?`python appshell/core/python_engine/engine_main.py` 鐨?`health` 鍔ㄤ綔杩斿洖 `status=ok`銆?  - 鍦?`.venv-win` 涓獙璇?`pytest tests/python_engine -q` 閫氳繃锛屾暣濂?Python 寮曟搸娴嬭瘯鍏?23 椤瑰叏閮ㄩ€氳繃銆?  - 鐢熸垚 `requirements.lock.txt`锛岄攣瀹氬綋鍓嶉€氳繃鍥炲綊鐨勫畬鏁翠緷璧栭泦鍚堛€?  - 鏂板 `scripts/setup_windows_env.ps1`锛屾敮鎸佸垱寤?閲嶅缓 `.venv-win` 骞跺彲閫夎嚜鍔ㄦ墽琛?Python 鍥炲綊銆?  - 鏂板 `ENVIRONMENT.md`锛屾槑纭鏄庣嫭绔嬬幆澧冭兘淇濊瘉浠€涔堬紝浠ュ強涓轰粈涔堢湡姝ｅ垎鍙戞椂浠嶉渶瑕佹墦鍖?Python runtime銆?  - 鏇存柊 `README.md` 涓?`appshell/README.md`锛屾妸鐙珛鐜鍏ュ彛绾冲叆浠撳簱榛樿浣跨敤鏂瑰紡銆?- 寰呭鐞嗕簨椤癸細
  - 璇勪及鏄惁灏?`.venv-appshell` 鏍囪涓洪仐鐣欐垨绉婚櫎锛岄伩鍏嶅悗缁鐢ㄣ€?  - 鍦ㄥ墠绔富鐣岄潰琛モ€滄煡鐪嬫渶杩戝惎鍔ㄨ瘖鏂€濆叆鍙ｃ€?  - 缁х画鎺ㄨ繘 CSV 瀵煎叆浣撻獙銆佷慨澶嶇粨鏋滅鐞嗗拰 Windows 鎵撳寘闂幆銆?
## Update 2026-03-12 11:18:58

- 鏀瑰姩鏃ユ湡锛?026-03-12 11:18:58
- 鏀瑰姩鍐呭绠€杩帮細淇 Anaconda base 鐜涓崯鍧忕殑 `numpy/pandas`锛屾仮澶?Python 寮曟搸鍋ュ悍妫€鏌ヤ笌鏁村 `tests/python_engine` 鍥炲綊锛涘悓鏃惰ˉ榻?`pytest` 鐩存帴鎵ц鏃剁殑瀵煎叆璺緞闂銆?- 鐩稿叧妯″潡/鏂囦欢锛?  - `tests/conftest.py`
  - `MEMO.md`
- 宸茶В鍐崇殑闂/鏂板鍔熻兘锛?  - 閫氳繃閲嶈 `numpy==1.26.4` 涓?`pandas==2.1.4` 淇浜?base 鐜涓?`numpy` 琚В鏋愪负 namespace package銆佺己灏?`__version__` 鐨勯棶棰樸€?  - 楠岃瘉 `python appshell/core/python_engine/engine_main.py` 鐨?`health` 鍔ㄤ綔宸叉仮澶?`status=ok`锛屽苟鑳芥纭繑鍥?`pandas / numpy / lightgbm / scikit-learn / joblib` 鐗堟湰淇℃伅銆?  - 楠岃瘉 `pytest tests/python_engine -q` 宸叉仮澶嶉€氳繃锛屾暣濂?Python 寮曟搸娴嬭瘯鍏?23 椤瑰叏閮ㄩ€氳繃銆?  - 鏂板 `tests/conftest.py`锛屽皢浠撳簱鏍圭洰褰曠ǔ瀹氬姞鍏?`sys.path`锛岄伩鍏?`pytest` 鍏ュ彛鑴氭湰鐩存帴杩愯鏃跺嚭鐜?`ModuleNotFoundError: src`銆?- 寰呭鐞嗕簨椤癸細
  - 璇勪及鏄惁闇€瑕佹妸 `base` 鐜缁х画鏀舵暃鍥炵函 conda 绠＄悊锛屾垨涓洪」鐩崟鐙垱寤轰笓鐢ㄨ櫄鎷熺幆澧冦€?  - 鍦ㄥ墠绔富鐣岄潰琛モ€滄煡鐪嬫渶杩戝惎鍔ㄨ瘖鏂€濆叆鍙ｃ€?  - 缁х画鎺ㄨ繘 CSV 瀵煎叆浣撻獙銆佷慨澶嶇粨鏋滅鐞嗗拰 Windows 鎵撳寘闂幆銆?
## Update 2026-03-12 10:55:50

- 鏀瑰姩鏃ユ湡锛?026-03-12 10:55:50
- 鏀瑰姩鍐呭绠€杩帮細瀹屾垚涓€杞祴璇曞洖褰掞紝骞舵妸涓婃鏈敹灏剧殑鏂囨。涓庣邯瑕佹暣鐞嗗共鍑€锛屾槑纭尯鍒嗏€滀唬鐮佸洖褰掆€濆拰鈥滄湰鏈虹幆澧冩崯鍧忊€濄€?- 鐩稿叧妯″潡/鏂囦欢锛?  - `MEMO.md`
  - `README.md`
  - `appshell/backend/README.md`
- 宸茶В鍐崇殑闂/鏂板鍔熻兘锛?  - 鍥炲綊楠岃瘉浜?`go test ./...`銆乣node --check appshell/frontend/src/main.js`銆乣pytest tests/python_engine/test_engine_health.py -q` 鍧囬€氳繃銆?  - 鍥炲綊纭鏁村 `pytest tests/python_engine -q` 澶辫触鐨勬牴鍥犱粛鏄湰鏈?`numpy/pandas` 鐜鎹熷潖锛岃€屼笉鏄湰娆′唬鐮佹敼鍔ㄥ紩鍏ョ殑鍔熻兘鍥為€€銆?  - 鏍圭洰褰?`README.md` 宸茶ˉ鍏呮闈㈢鍚姩鑷璇存槑锛屽寘鍚鏌ヨ寖鍥淬€侀樆濉炶鍒欏拰娴忚鍣?mock 璇存槑銆?  - `appshell/backend/README.md` 宸茶ˉ鍏?`RunStartupChecks()`銆佸惎鍔ㄦ嫤鎴涓哄拰鎶ュ憡缁撴瀯璇存槑銆?  - `MEMO.md` 宸查噸鍐欎负娓呮櫚鍙淮鎶ょ殑椤圭洰绾锛屼繚鐣欐渶缁堢洰鏍囥€佹柟娉曘€佸凡瀹屾垚姝ラ銆佸綋鍓嶉棶棰樺拰杩戞湡鏇存柊銆?- 寰呭鐞嗕簨椤癸細
  - 淇鎴栭噸寤哄綋鍓?Python 杩愯鐜锛屽啀鎭㈠ `tests/python_engine` 鍏ㄩ噺鍥炲綊銆?  - 鍦ㄥ墠绔富鐣岄潰琛モ€滄煡鐪嬫渶杩戝惎鍔ㄨ瘖鏂€濆叆鍙ｃ€?  - 缁х画鎺ㄨ繘 CSV 瀵煎叆浣撻獙銆佷慨澶嶇粨鏋滅鐞嗗拰 Windows 鎵撳寘闂幆銆?
## Update 2026-03-12 10:38:47

- 鏀瑰姩鏃ユ湡锛?026-03-12 10:38:47
- 鏀瑰姩鍐呭绠€杩帮細涓?`appshell` 澧炲姞鍚姩鑷 v1锛屾柊澧為樆濉炲紡 preflight gate锛屽苟鎶?Python `health` 鍗囩骇涓虹湡瀹炶繍琛屾椂渚濊禆璇婃柇銆?- 鐩稿叧妯″潡/鏂囦欢锛?  - `appshell/core/python_engine/engine_core.py`
  - `tests/python_engine/test_engine_health.py`
  - `appshell/backend/cmd/wails/app.go`
  - `appshell/backend/cmd/wails/startup_checks.go`
  - `appshell/backend/cmd/wails/startup_checks_test.go`
  - `appshell/frontend/index.html`
  - `appshell/frontend/src/main.js`
  - `appshell/frontend/src/style.css`
  - `MEMO.md`
- 宸茶В鍐崇殑闂/鏂板鍔熻兘锛?  - 鏂板 Go 渚?`RunStartupChecks()`锛岃仛鍚?`engine_script`銆乣engine_health`銆乣runtime_dependencies`銆乣task_history_sqlite`銆乣results_output_root`銆乣model_artifacts` 鍏被妫€鏌ョ粨鏋溿€?  - 鍚姩闃舵鏀逛负鈥滃厛鑷銆佸悗鍒濆鍖?service鈥濓紝鑷鏈€氳繃鏃?`RunTask / GetTaskStatus / CancelTask / ListTaskHistory` 浼氱粺涓€杩斿洖鍚姩鎷︽埅閿欒銆?  - Python `health` 鐜板湪閫愰」妫€鏌?`pandas / numpy / lightgbm / scikit-learn / joblib`锛岀己澶辨垨鎹熷潖鏃惰繑鍥炵粨鏋勫寲 `MISSING_DEPENDENCY`銆?  - 鍓嶇鏂板闃诲寮忓惎鍔ㄨ嚜妫€棣栧睆锛屾敮鎸佽鎯呭睍寮€銆佸鍒惰瘖鏂俊鎭拰澶辫触鍚庨噸璇曪紱娴忚鍣ㄩ潤鎬侀瑙堟ā寮忎笅鎻愪緵 mock 缁撴灉銆?  - 鏂板 Python 鍗曟祴 `test_engine_health.py` 鍜?Go 鑷娴嬭瘯 `startup_checks_test.go`銆?- 寰呭鐞嗕簨椤癸細
  - 鍦ㄤ富鐣岄潰琛モ€滄煡鐪嬫渶杩戝惎鍔ㄨ瘖鏂€濆叆鍙ｃ€?  - 缁х画琛?CSV 瀵煎叆浣撻獙鍜屼慨澶嶇粨鏋滅鐞嗐€?  - 鍚庣画鑰冭檻缁熶竴榛樿妯″瀷鐩綍绛栫暐锛屽噺灏?`data/processed` 涓?`outputs/results/*` 鍙岃矾寰勫甫鏉ョ殑鐞嗚В鎴愭湰銆?
## Update 2026-03-12 08:57:33

- 鏀瑰姩鏃ユ湡锛?026-03-12 08:57:33
- 鏀瑰姩鍐呭绠€杩帮細涓洪椤垫柊澧炰竴濂椾笓闂ㄩ潰鍚?Figma 璁捐浜や粯鐨勯潤鎬佽璁℃澘锛屼笉鍐嶇洿鎺ヤ緷璧栫幇鏈夎繍琛岄〉鍋氱綉椤靛鍏ャ€?- 鐩稿叧妯″潡/鏂囦欢锛?  - `appshell/frontend/figma-home-native.html`
  - `appshell/frontend/src/figma-home-native.css`
  - `MEMO.md`
- 宸茶В鍐崇殑闂/鏂板鍔熻兘锛?  - 鏂板鐢ㄤ簬鐢熸垚 Figma 璁捐鏂囦欢鐨勯椤佃璁℃簮椤甸潰銆?  - 灏嗛椤典富瑙嗚銆佸伐浣滄祦姝ラ銆侀厤缃尯銆佽瘖鏂晶鏍忓拰缁勪欢灞曠ず鍖烘媶鎴愰潤鎬佸彲鎹曡幏缁撴瀯锛岄檷浣?Figma 瀵煎叆閫€鍖栭闄┿€?  - 淇濈暀鍘熸湁浜у搧鎬濊矾锛屼絾涓嶅啀鍙楅檺浜庡綋鍓嶈繍琛屾€?UI 鐨勫竷灞€鍜屼氦浜掑疄鐜般€?- 寰呭鐞嗕簨椤癸細
  - 浣跨敤鏂拌璁℃澘鐢熸垚鏂扮殑 Figma 鏂囦欢骞堕獙璇佸鍏ヨ川閲忋€?  - 濡備粛鏈夌粨鏋勬崯澶憋紝鑰冭檻鎶婂叧閿尯鍩熸媶鎴愬寮犻〉闈㈠垎鍒鍏ャ€?  - 瑙嗛渶瑕佸啀琛ヤ竴涓€滄娴嬬粨鏋滄€佲€濊璁℃澘銆?
## Baseline Summary

- `src/training_core.py` 涓?`src/repair_core.py` 宸叉壙鎷呮牳蹇冭缁冧笌淇閫昏緫銆?- `appshell/core/python_engine/` 宸插舰鎴愬崗璁寲杈圭晫锛欳LI 鍏ュ彛銆佸姩浣滆矾鐢便€侀敊璇粨鏋勩€佹棩蹇楀拰鏍稿績涓氬姟閫昏緫鍒嗗眰鏄庣‘銆?- `appshell/backend/internal/task/` 宸插叿澶囦换鍔＄姸鎬佹祦杞€丼QLite 鍘嗗彶璁板綍鍜屾渶杩戜换鍔℃煡璇㈣兘鍔涖€?- `appshell/frontend/src/main.js` 宸插洿缁曞洓姝ュ伐浣滄祦瀹炵幇妗岄潰绔富娴佺▼锛屽苟涓庣湡瀹?Wails 缁戝畾鑱旈€氥€?- `outputs/` 涓?`thesis-defense/` 缁х画鎵挎媴缁撴灉浜х墿鍜岀瓟杈╂潗鏂欐矇娣€鑱岃矗銆?

## Update 2026-03-13 15:57:35

- 鏀瑰姩鏃ユ湡锛?026-03-13 15:57:35
- 鏀瑰姩鍐呭绠€杩帮細鏍规嵁 Go 鍚庣姹傝亴鏂瑰悜锛岄噸鍐欐湰椤圭洰鐨勭畝鍘嗛」鐩粡鍘嗭紝绐佸嚭 Go 缂栨帓 Python 瀛愯繘绋嬬殑璺ㄨ瑷€鏋舵瀯銆佹湰鍦?CSV 鏁版嵁璐ㄦ闂幆涓庡伐绋嬪寲鑳藉姏锛屽幓闄?make target銆佺畻娉曟寚鏍囧拰鈥滄祴璇曢€氳繃鈥濆紡寮辫〃杈俱€?- 鐩稿叧妯″潡/鏂囦欢锛?  - `resume_project_entry.tex`
  - `MEMO.md`
- 宸茶В鍐崇殑闂/鏂板鍔熻兘锛?  - 鏂板鍙洿鎺ョ矘璐村埌 LaTeX 绠€鍘嗕腑鐨勯」鐩潯鐩紝涓诲彊浜嬫敼涓衡€滄湰鍦版暟鎹川妫€涓庝慨澶嶅钩鍙扳€濄€?  - 棣栨潯鎻忚堪鏄庣‘璇存槑閲囩敤 Python 璐熻矗鏁版嵁澶勭悊寮曟搸銆丟o 璐熻矗浠诲姟鐢熷懡鍛ㄦ湡鍜屽苟鍙戣皟搴︺€乄ails 璐熻矗妗岄潰浜や簰鐨勫垎灞傝璁°€?  - 鐢ㄥ綋鍓嶄粨搴撳凡钀藉湴鐨勭‖浜嬪疄鏇夸唬寮辫〃杩帮細榛樿骞跺彂 3銆佸彇娑?2 绉掑唴鐢熸晥銆佽秴鏃跺悗鏃犲兊灏歌繘绋嬨€丟o 渚?25 涓祴璇曞叆鍙ｃ€丳ython 寮曟搸 23 涓敤渚嬨€?  - 鍒犻櫎涓嶇鍚堝綋鍓嶅疄鐜扮姸鎬佺殑琛ㄨ堪锛屽鈥滀竴閿儴缃测€濃€滃畬鏁村畨瑁呭櫒浜や粯鈥濃€滆鐩栫巼鏁板瓧鈥濃€渕ake up/health/demo/test 涓€閿獙鏀垛€濄€?- 寰呭鐞嗕簨椤癸細
  - 鍚庣画鍙牴鎹洰鏍囧矖浣嶅啀鍑轰竴鐗堚€滄洿鍋忓熀纭€鏋舵瀯鈥濇垨鈥滄洿鍋忎笟鍔″悗绔€濈殑鍘嬬缉鐗堥」鐩弿杩般€?  - 鑻ョ畝鍘嗘暣浣撶瘒骞呭彈闄愶紝鍙繘涓€姝ユ妸绗?4銆? 鏉″帇缂╁悎骞朵负 1 鏉°€?
## Update 2026-03-16 20:58:19

- 鏀瑰姩鏃ユ湡锛?026-03-16 20:58:19
- 鏀瑰姩鍐呭绠€杩帮細瀹屾垚 `MULTI_AGENT_BLUEPRINT.md` 涓殑 Stage 5锛屾妸 Wails 涓讳綋楠屼粠鍥涙鍚戝浼樺厛鏀舵暃涓衡€滈€夋嫨鏂囦欢 -> 鏌ョ湅鎽樿 -> 涓€閿櫤鑳藉鐞?-> 鏌ョ湅缁撴灉涓庡璁♀€濈殑榛樿娴侊紝鍚屾椂淇濈暀楂樼骇宸ヤ綔鍙版壙杞界粡鍏?scan/manual repair/璋冭瘯璺緞銆?- 鐩稿叧妯″潡/鏂囦欢锛?  - `appshell/frontend/index.html`
  - `appshell/frontend/src/main.js`
  - `appshell/frontend/src/style.css`
  - `MEMO.md`
- 宸茶В鍐崇殑闂/鏂板鍔熻兘锛?  - 榛樿棣栭〉鐜板湪鐩存帴杩涘叆 `smart_home`锛屼紭鍏堝睍绀轰竴閿笂浼犲叆鍙ｃ€侀鍚姩鎽樿鍗″拰榛樿瀹夊叏杈圭晫锛屼笉鍐嶅厛鏆撮湶鍥涙鍙傛暟鍚戝銆?  - 鏂板 `smart_home / smart_run / smart_result / advanced_workspace` 瑙嗗浘绾х粨鏋勶紝骞朵繚鐣欓珮绾у伐浣滃彴浣滀负缁忓吀娴佺▼鍏ュ彛銆?  - 鍓嶇宸叉寮忔帴鍏?`RunAgentAutofixSession` 浣滀负榛樿鎵ц閾捐矾锛涢€夋嫨鏂囦欢鍚庡厛閫氳繃 `ListCSVColumns` 鐢熸垚鎽樿锛屽啀鐢辩敤鎴风‘璁ゅ悗鍚姩鏅鸿兘闂幆銆?  - 琛ラ綈鏈€杩戜换鍔℃仮澶嶉€昏緫锛氫紭鍏堟仮澶嶈繍琛屼腑鐨?`agent.session.auto`锛屽叾娆℃仮澶嶆渶杩戝畬鎴愮殑鏅鸿兘浠诲姟锛屽叾浣欎换鍔′粛鍥為€€鍒伴珮绾у伐浣滃彴銆?  - 鏂板缁撴灉椤靛彲淇″澹筹細safety banner銆佸叧閿骇鐗╁叆鍙ｃ€佹墽琛岀悊鐢便€佽建杩逛笌楠岃瘉銆佸師濮嬬粨鏋滄娊灞夛紝缁熶竴鎵挎帴 Stage 4 鐨?`presentation`銆?  - 鍚姩鑷鏂板鎸佷箙鍏ュ彛 `鏌ョ湅鍚姩璇婃柇`锛岃繘鍏ュ簲鐢ㄥ悗浠嶅彲閲嶆柊鎵撳紑鐜版湁 startup report銆?  - 琛ラ綈鏅鸿兘妯″紡涓嬬殑鎷栨嫿/閫夋嫨 CSV銆佽緭鍑虹洰褰曞悓姝ャ€佸彇娑堜换鍔°€佸鍑?JSON/CSV銆佽繑鍥炴櫤鑳芥ā寮忕瓑浜嬩欢缁戝畾鍜屾牱寮忓眰銆?  - 缁存寔鏃ц兘鍔涗笉鍥為€€锛氶珮绾у伐浣滃彴涓殑缁忓吀 scan/manual repair 閾捐矾浠嶄繚鐣欙紝鍚庣 API 鍜?Python stable actions 鏈柊澧炵牬鍧忔€у彉鍖栥€?  - 鏈疆鍥炲綊閫氳繃锛?    - `node --check appshell/frontend/src/main.js`
    - `go test ./...`锛坄appshell/backend`锛?    - `python -m pytest tests/python_engine -q`锛?9 passed锛?- 寰呭鐞嗕簨椤癸細
  - 鍋氫竴杞湡瀹?Wails 妗岄潰绔汉宸ラ獙鏀讹紝閲嶇偣楠岃瘉榛樿棣栭〉銆佹渶杩戜换鍔℃仮澶嶃€佹櫤鑳界粨鏋滈〉鎶藉眽鍜岄珮绾у伐浣滃彴寰€杩斿垏鎹㈢殑瀹為檯浜や簰銆?  - 璇勪及鏄惁灏?Stage 5 鐨勬櫤鑳芥ā寮忚ˉ鍏呭埌 README / 浣跨敤璇存槑涓紝甯姪棣栨浣跨敤鑰呯悊瑙ｂ€滈粯璁や竴閿ā寮忊€濅笌鈥滈珮绾у伐浣滃彴鈥濈殑鍒嗗伐銆?  - 缁х画瑙傚療鏄惁闇€瑕佹妸鏅鸿兘妯″紡缁撴灉椤典腑鐨勯儴鍒?artifact 鍋氭垚鍙偣鍑绘墦寮€鎴栧鍒惰矾寰勭殑澧炲己浜や簰銆?
## Update 2026-03-16 21:28:38

- 鏀瑰姩鏃ユ湡锛?026-03-16 21:28:38
- 鏀瑰姩鍐呭绠€杩帮細鏂板鏍圭洰褰曚笓椤硅矾绾垮浘 `LANGGRAPH_UPGRADE_ROADMAP.md`锛屾槑纭鏋滄湭鏉ュ紩鍏?LangGraph锛屽簲璇ュ浣曞湪涓嶆帹缈荤幇鏈?Go + Python + Wails 鏋舵瀯鐨勫墠鎻愪笅锛屾妸 LangGraph 浣滀负璁ょ煡灞?sidecar 娓愯繘鎺ュ叆锛岃€屼笉鏄浛浠ｅ綋鍓?deterministic 鎵ц浣撶郴銆?- 鐩稿叧妯″潡/鏂囦欢锛?  - `LANGGRAPH_UPGRADE_ROADMAP.md`
  - `MEMO.md`
- 宸茶В鍐崇殑闂/鏂板鍔熻兘锛?  - 鏂板 LangGraph 涓撻」鍗囩骇鏂囨。锛屼綔涓?`MULTI_AGENT_BLUEPRINT.md` 鐨勮ˉ鍏呰矾绾垮浘锛岃仛鐒︹€滄槸鍚︽帴鍏ャ€佹帴鍦ㄥ摢涓€灞傘€佷负浠€涔堣繖鏍锋帴鈥濄€?  - 姝ｅ紡鍐荤粨 LangGraph 鐨勫畾浣嶏細Stage 5 涔嬪悗鐨勫彲閫夎鐭ュ眰鍗囩骇璺嚎锛岃€屼笉鏄鐜版湁 deterministic multi-agent/runtime/tool layer 鐨勬浛浠ｃ€?  - 鏄庣‘鍝簺妯″潡缁х画淇濇寔 deterministic锛歚scan_file`銆乣repair_batch`銆乣repair_with_gower`銆乣rollback_repair_batch`銆乿alidation銆乺escan銆乺ollback gate銆?  - 鏄庣‘鍝簺妯″潡閫傚悎鍚庣画鎺ュ叆 LangGraph锛歚Intent / Strategy / Explainer / Approval / Preference-Memory`锛屽苟寮鸿皟鍙仛鐭緭鍏ャ€佺煭杈撳嚭銆佸己缁撴瀯鍖栧缓璁€?  - 姝ｅ紡鍐欏畾鏈潵鎺ㄨ崘鏋舵瀯锛歚Wails Frontend + Go Control Plane + Python Deterministic Tool Layer + Python LangGraph Sidecar`銆?  - 姝ｅ紡鍐欏畾鏈潵 sidecar 閫氫俊寤鸿锛氭湰鍦?loopback HTTP/JSON锛岄鐣?`GET /health`銆乣POST /v1/plan`銆乣POST /v1/explain`銆乣POST /v1/approve` 鍥涚被鎺ュ彛銆?  - 鏄庣‘涓嶉噰鐢ㄧ殑鏂瑰悜锛氫笉鎶婂綋鍓嶇郴缁熸敼鎴?LangGraph-first銆佷笉璁?LangGraph 鐩存帴鍐欐枃浠躲€佷笉鎶?validation/rollback 浜ょ粰 LLM銆佷笉鎶?UI 鍙樻垚闀?prompt 杈撳叆鍣ㄣ€佷笉鎶婄粨鏋滈〉鍙樻垚闀跨瘒 AI 鎶ュ憡銆?  - 鏂囨。涓凡缁欏嚭涓ゅ紶 Mermaid 鍥撅紝鍒嗗埆璇存槑鐩爣鏋舵瀯涓庡垎闃舵鍗囩骇璺嚎锛屼究浜庡悗缁伐绋嬪疄鐜板拰绛旇京璁茶В澶嶇敤銆?- 寰呭鐞嗕簨椤癸細
  - 鍚庣画鑻ョ湡姝ｈ繘鍏?LangGraph 瀹炴柦闃舵锛屽厛鍦?Go 渚цˉ `LangGraphClient / LangGraphPlanner` 鎶借薄锛屽啀寮曞叆鏈€灏?Python sidecar skeleton銆?  - 鍦ㄤ笉鏀瑰彉鐜版湁涓€閿棴鐜綋楠岀殑鍓嶆彁涓嬶紝璇勪及鍝竴杞渶閫傚悎鍏堟浛鎹?`MockPlanner` 鐨?`Intent / Strategy / Explain` 璁ょ煡閮ㄥ垎銆?  - 鑻ュ悗缁喅瀹氭帴鐪熷疄 LLM API锛岄渶瑕佸崟鐙ˉ涓€浠?provider 绛栫暐鏂囨。锛屾槑纭?OpenAI / 鏈湴妯″瀷 / 鍏朵粬渚涘簲鍟嗙殑鎺ュ叆杈圭晫涓庨檷绾х瓥鐣ャ€?
## Update 2026-03-16 22:00:19

- 鏀瑰姩鏃ユ湡锛?026-03-16 22:00:19
- 鏀瑰姩鍐呭绠€杩帮細瀹屾垚 `LANGGRAPH_UPGRADE_ROADMAP.md` 鐨?Phase A锛屼互杞婚噺鍐荤粨鏂瑰紡鏀剁揣 `Planner` 涓?Go control plane 鐨勮竟鐣岋紝涓嶆敼鐜版湁琛屼负銆佷笉鍔?deterministic tool layer銆佷笉鏀?wire protocol锛屽彧閫氳繃鎶借薄褰掍綅銆佹敞閲婂拰娴嬭瘯鎶娾€滄湭鏉?LangGraph 浠呮浛鎹?Planner 璁ょ煡瀹炵幇鈥濆浐瀹氫笅鏉ャ€?- 鐩稿叧妯″潡/鏂囦欢锛?  - `appshell/backend/internal/agent/planner.go`
  - `appshell/backend/internal/agent/planning_support.go`
  - `appshell/backend/internal/agent/planning_flow.go`
  - `appshell/backend/internal/agent/mock_planner.go`
  - `appshell/backend/internal/agent/mock_planner_test.go`
  - `appshell/backend/internal/agent/planning_support_test.go`
  - `appshell/backend/internal/agent/runtime_runner_test.go`
  - `MEMO.md`
- 宸茶В鍐崇殑闂/鏂板鍔熻兘锛?  - 涓?`PlanningInput` 鍜?`Planner` 澧炲姞杈圭晫娉ㄩ噴锛屾槑纭畠浠睘浜庤鐭ュ眰鎺ュ彛锛歱lanner 鍙秷璐?runtime 棰勫厛缁勮濂界殑 deterministic snapshot锛屽彧杩斿洖 `AgentPlan`锛屼笉璐熻矗 tool calling銆乸ersistence銆乿alidation銆乺ollback 鎴栨枃浠跺啓鍏ャ€?  - 鏂板涓€?helper `planning_support.go`锛屾妸 `supportedRepairIssueTypes`銆乣selectRepairableIssues` 鍜?`buildPlanningInput` 浠?`MockPlanner` 鏂囦欢涓Щ鍑猴紝閬垮厤 `planning_flow.go` 缁х画渚濊禆 planner 绉佹湁瀹炵幇銆?  - `planning_flow.go` 鐜板湪閫氳繃 `buildPlanningInput(...)` 鎶?scan 缁撴灉銆乺ule/gower preview銆乮ssue 閫夋嫨缁撴灉鍜?overrides 缁熶竴鍏嬮殕鍚庡啀浼犵粰 `Planner.BuildPlan()`锛岃繘涓€姝ュ浐瀹氫簡鈥淕o runtime 璐熻矗 deterministic preparation锛宲lanner 鍙悆缁撴灉鈥濈殑鑱岃矗鍒掑垎銆?  - 涓?`MockPlanner` 澧炲姞缂栬瘧鏈熸帴鍙ｆ柇瑷€锛屾槑纭畠鍙槸褰撳墠 `Planner` 鐨勯粯璁ゅ疄鐜帮紝鑰屼笉鏄叡浜?deterministic 鍑嗗閫昏緫鐨勬壙杞界偣銆?  - 娴嬭瘯灞傛柊澧?Phase A 杈圭晫淇濇姢锛?    - `planning_support_test.go` 楠岃瘉 repairable issue 閫夋嫨閫昏緫鍜?`PlanningInput` 鍏嬮殕璇箟锛?    - `runtime_runner_test.go` 鏂板 `spyPlanner`锛岄獙璇?runtime 浼氬厛瀹屾垚 scan 鍜?rule/gower preview锛屽啀鎶婂畬鏁村揩鐓т氦缁?planner锛?    - `mock_planner_test.go` 鑱氱劍 `MockPlanner` 鍩轰簬 deterministic previews 鏋勫缓 `rule / gower / hybrid` 涓夊€欓€夌殑璁ょ煡琛屼负銆?  - 鏈疆鍥炲綊閫氳繃锛?    - `go test ./...`锛坄appshell/backend`锛?    - `python -m pytest tests/python_engine -q`锛?9 passed锛?- 寰呭鐞嗕簨椤癸細
  - 缁х画鎸?`LANGGRAPH_UPGRADE_ROADMAP.md` 鎺ㄨ繘 Phase B锛屽湪涓嶆墦鐮村綋鍓嶈竟鐣岀殑鍓嶆彁涓嬪紩鍏ユ渶灏?`LangGraph` sidecar skeleton銆?  - 鍚庣画鑻ユ浛鎹?`MockPlanner`锛屽簲浼樺厛澶嶇敤鏈疆鍐荤粨鍚庣殑 `Planner` 鍚堝悓鍜?`buildPlanningInput(...)` 蹇収杈圭晫锛屼笉鍐嶆妸 deterministic 鍑嗗閫昏緫甯﹀洖 planner 瀹炵幇鍐呴儴銆?  - 濡傚悗缁紩鍏?`LangGraphClient / LangGraphPlanner`锛岄渶瑕佸悓姝ヨ瘎浼?trace 鏄犲皠鍜岄檷绾у洖閫€鍒扮幇鏈?`MockPlanner` 鐨勫け璐ョ瓥鐣ャ€?`r`n`r`n## Update 2026-03-17 10:41:00

- 鏀瑰姩鏃ユ湡锛?026-03-17 10:41:00
- 鏀瑰姩鍐呭绠€杩帮細瀹屾垚 `LANGGRAPH_UPGRADE_ROADMAP.md` Phase B 绗竴杞惤鍦帮紝鍦ㄤ笉鏀?deterministic tool layer銆佷笉鎺ョ湡瀹?LLM 鐨勫墠鎻愪笅锛屽紩鍏モ€淕o 鑷姩鎷夎捣 + Python 鏈湴 LangGraph sidecar + 鑷姩 fallback鈥濈殑鏈€灏忛鏋躲€?- 鐩稿叧妯″潡/鏂囦欢锛?  - `appshell/core/langgraph_sidecar/__init__.py`
  - `appshell/core/langgraph_sidecar/main.py`
  - `appshell/core/langgraph_sidecar/server.py`
  - `appshell/core/langgraph_sidecar/graph.py`
  - `appshell/core/langgraph_sidecar/schemas.py`
  - `appshell/backend/internal/agent/langgraph_types.go`
  - `appshell/backend/internal/agent/langgraph_client.go`
  - `appshell/backend/internal/agent/langgraph_manager.go`
  - `appshell/backend/internal/agent/langgraph_planner.go`
  - `appshell/backend/internal/agent/langgraph_factory.go`
  - `appshell/backend/internal/agent/langgraph_client_test.go`
  - `appshell/backend/internal/agent/langgraph_manager_test.go`
  - `appshell/backend/internal/agent/langgraph_planner_test.go`
  - `appshell/backend/cmd/wails/startup_checks.go`
  - `appshell/backend/cmd/wails/startup_checks_test.go`
  - `appshell/backend/cmd/wails/app.go`
  - `appshell/backend/cmd/demo/main.go`
  - `tests/langgraph_sidecar/test_graph.py`
  - `tests/langgraph_sidecar/test_server.py`
  - `requirements.txt`
  - `requirements.lock.txt`
  - `ENVIRONMENT.md`
  - `scripts/setup_windows_env.ps1`
  - `MEMO.md`
- 宸茶В鍐崇殑闂/鏂板鍔熻兘锛?  - 鏂板鏈湴 Python sidecar锛屾毚闇?`GET /health` 涓?`POST /v1/plan`锛屽苟閫氳繃 LangGraph `StateGraph` 鍗曡妭鐐圭┖鍥捐繑鍥?mock planning 缁撴灉銆?  - Go 渚ф柊澧?`LangGraphConfig / LangGraphClient / LangGraphSidecarManager / LangGraphPlanner`锛岄€氳繃 decorator 鏂瑰紡鍖呰鐜版湁 `MockPlanner`銆?  - `LangGraphPlanner` 鍏堢敓鎴愬畬鏁?deterministic `AgentPlan`锛屽啀灏濊瘯璋冪敤 sidecar 瑕嗙洊璁ょ煡瀛楁锛泂idecar 涓嶅彲鐢ㄦ椂鑷姩鍥為€€锛屼笉涓柇 `agent.session.plan`銆?  - Wails startup checks 鏂板 `langgraph_sidecar` 闈為樆濉炴鏌ラ」锛屼細灏濊瘯棰勭儹 sidecar锛涘け璐ュ彧璁?`warning`锛屼笉闃诲涓绘祦绋嬨€?  - Wails service 涓?demo CLI 宸叉帴鍏ュ悓涓€濂?planner 鏍堬紝鍏抽棴 service/app 鏃朵細鏄惧紡鍏抽棴鐢?Go 鎷夎捣鐨?sidecar 瀛愯繘绋嬨€?  - 鏂板 Go 娴嬭瘯瑕嗙洊 client 瑙ｆ瀽銆乻idecar 鑷姩鎷夎捣/閲嶅惎銆佺鍙ｅ崰鐢ㄥ洖閫€銆乸lanner overlay/fallback銆乻tartup checks pass/warning銆?  - 鏂板 Python 娴嬭瘯瑕嗙洊绌?graph 鍙?invoke銆乻idecar `/health` 涓?`/v1/plan` 鍙搷搴斻€?  - 鐜鍩虹嚎宸插姞鍏?`langgraph==1.1.2` 鍙婂叾閿佸畾渚濊禆锛宍.venv-win` 鍒濆鍖栬剼鏈細鍚屾椂璺?`tests/python_engine` 涓?`tests/langgraph_sidecar`銆?- 褰撳墠鏂规硶锛?  - Go control plane 缁х画璐熻矗 tool calling銆乿alidation銆乺ollback銆乻ession/trace銆?  - Python deterministic tool layer 淇濇寔涓嶅彉銆?  - LangGraph sidecar 浠呮彁渚?mock cognition JSON 鎺ュ彛锛屼笉鐩存帴鍐?CSV銆佷笉鎺ョ鍥炴粴銆佷笉鏇夸唬 `RuntimeRunner`銆?- 宸插畬鎴愰獙璇侊細
  - `go test ./...`锛坄appshell/backend`锛夐€氳繃銆?  - `.\\.venv-win\\Scripts\\python.exe -m pytest tests/python_engine tests/langgraph_sidecar -q` 閫氳繃锛屽綋鍓嶄负 `32 passed`銆?- 寰呭鐞嗕簨椤癸細
  - 鎺ㄨ繘 Phase C锛岃 LangGraph sidecar 寮€濮嬫浛鎹?`Intent + Strategy + Explain` 鐨勮鐭ュ疄鐜帮紝浣嗙户缁敱 Go runtime 鎵ц scan / repair / validate / rollback銆?  - 缁х画琛ュ厖 sidecar 澶辫触鍘熷洜鐨?trace / startup report 缁嗙矑搴︽槧灏勶紝渚嬪鍖哄垎 disabled銆乻cript missing銆乭ealth failed銆乸ort occupied銆?  - 鍦ㄦ湭鏉ョ湡瀹炴帴鍏?LLM API 鍓嶏紝缁х画鍧氭寔鈥滅煭杈撳叆銆佺煭杈撳嚭銆佸己缁撴瀯鍖栤€濈殑浜у搧绾︽潫銆?

## Update 2026-03-17 10:37:47

- 改动日期：2026-03-17 10:37:47
- 改动内容简述：为 LangGraph Phase C 增加可直接复用的本地 LLM 配置入口，按用户提供的 OpenAI-compatible 站点完成本地 PowerShell 配置；同时修正 sidecar 对该类网关的请求兼容性，并实测确认当前可用模型应为 `deepseek-v3.2` 而不是 `deepseek-v3.2 chat`。
- 相关模块/文件：
  - `.gitignore`
  - `scripts/langgraph.local.example.ps1`
  - `scripts/langgraph.local.ps1`
  - `scripts/run_wails_langgraph.ps1`
  - `appshell/core/langgraph_sidecar/llm_client.py`
  - `MEMO.md`
- 已解决的问题/新增功能：
  - 新增被 Git 忽略的本地 LangGraph 配置文件 `scripts/langgraph.local.ps1`，已写入用户提供的 API 站点、API Key、模型名和 `.venv-win` Python 路径。
  - 新增配置模板 `scripts/langgraph.local.example.ps1`，便于后续更换提供商或迁移到其他机器时复用。
  - 新增一键启动脚本 `scripts/run_wails_langgraph.ps1`，会先加载本地配置，再进入 `appshell/` 执行 `wails dev`。
  - `llm_client.py` 现已补充 `Accept` 和自定义 `User-Agent` 请求头，避免默认 `Python-urllib` 请求被部分 OpenAI-compatible 网关直接拦截。
  - 实测验证通过：`.\.venv-win\Scripts\python.exe -m pytest tests/langgraph_sidecar -q` 通过；真实最小 LLM 请求已成功返回 JSON。
- 待处理事项：
  - 用 `scripts/run_wails_langgraph.ps1` 做一轮真实 Wails 手工联调，确认 UI 中的 LangGraph planner 已进入 `llm` 模式而不是 fallback。
  - 如后续需要切换其他模型，优先先调用 `/models` 核对 provider 当前暴露的真实模型 ID，再更新 `scripts/langgraph.local.ps1`。
  - 如后续需要跨机器复用这套配置，考虑补一个不含密钥的 README/ENVIRONMENT 使用片段，避免再次手动排查模型名或请求头兼容性问题。

## Update 2026-03-17 16:18:54

- 改动日期：2026-03-17 16:18:54
- 改动内容简述：继续落实 `LANGGRAPH_UPGRADE_ROADMAP.md` 的 Phase D“Interrupt 与 Memory”，把审批门、工作区偏好和 smart 模式前端收口为完整的“两段式任务”体验，并补齐对应的 Go 回归测试与 Wails 方法测试。
- 最终目标：让默认智能闭环在“规划完成 + preview validation 通过之后、真正写文件之前”统一进入高风险审批点；同时把会话 memory、工作区默认偏好和前端审批/偏好卡串成可恢复、可审计、可保存默认设置的产品闭环。
- 当前采用的方法：
  - 后端继续由 Go 作为唯一真相源保存 session context 与 workspace preference profile，LangGraph 只消费整理后的 `preference_snapshot / approval_context / safety_context`。
  - smart 首页把“本次运行偏好草稿”和“保存为工作区默认”分离；smart 结果页把审批请求、继续执行与取消执行整合进同一结果面板。
  - 测试层优先补高风险审批暂停、审批通过恢复、审批拒绝不写文件、时间列审批、偏好持久化与 Wails 暴露方法，确保 Phase D 不回退已有 auto/rollback 闭环。
- 相关模块/文件：
  - `appshell/frontend/index.html`
  - `appshell/frontend/src/main.js`
  - `appshell/frontend/src/style.css`
  - `appshell/backend/internal/agent/runtime_runner_test.go`
  - `appshell/backend/internal/agent/sqlite_store_test.go`
  - `appshell/backend/cmd/wails/app_test.go`
  - `MEMO.md`
- 已完成的步骤 / 已解决的问题 / 新增功能：
  - smart 首页新增“修复偏好”卡，支持加载当前工作区默认偏好、编辑 `conservative_mode / avoid_time_columns / protected_columns / require_approval_for_high_risk`，并通过 `SaveAgentPreferences` 显式持久化。
  - smart 模式启动任务时，前端现在会把当前偏好草稿作为 `user_preferences` 连同 `workspace_id` 一起发送给 `RunAgentAutofixSession`，与 Phase D 的偏好合并顺序保持一致。
  - smart 结果页新增审批卡，展示审批状态、候选来源、影响列、触发原因和记忆中的偏好快照；在 `approval_required` 状态下只暴露“继续执行”和“取消本次执行”两个动作。
  - 前端新增 `ApproveAgentSession / GetAgentPreferences / SaveAgentPreferences` 的真实 binding 与 preview-mode mock，实现预览模式下也能演示完整审批恢复流程。
  - preview-mode mock 现在支持：
    - 工作区偏好读写
    - `approval_required -> approve -> accepted`
    - `approval_required -> reject -> approval_rejected`
    - 结果页与 trace 的审批状态刷新
  - 修复 smart 结果页里 `renderSmartReasoning(...)` 对未定义变量 `explanationBlock` 的运行时错误，避免结果页在渲染 reasoning 时崩掉。
  - Go 回归新增并通过：
    - SQLite 偏好保存 / 覆盖 / 重启后读取测试
    - auto 会话命中高风险列后进入 `awaiting_approval`
    - `agent.session.approve(approve)` 恢复 auto 执行并完成
    - `agent.session.approve(reject)` 不写输出、不回滚、会话进入 `approval_rejected`
    - execute 路径命中时间列后同样走审批门
    - Wails `GetAgentPreferences / SaveAgentPreferences / ApproveAgentSession` 方法测试
- 已完成验证：
  - `go test ./...`（`appshell/backend`）通过
  - `node --check appshell/frontend/src/main.js` 通过
- 当前问题 / 待处理事项：
  - 还需要做一轮真实 Wails 桌面端手工联调，重点确认 smart 首页偏好卡、审批卡、继续执行/取消执行按钮和最近任务恢复在真实 binding 下的交互细节。
  - 目前前端 preview mock 只模拟了最常见的审批路径，若后续要演示更多风险场景，可以再补 protected/time/high-risk 的更细分 mock 数据。
  - Phase D 已经把审批与 memory 闭环打通，但 README / 使用说明里还没有同步更新“两段式任务”和“工作区默认偏好”的产品说明。

## Update 2026-03-17 17:27:37

- 改动日期：2026-03-17 17:27:37
- 改动内容简述：完成 `LANGGRAPH_UPGRADE_ROADMAP.md` 的 Phase E“稳定化与产品化”，把 LangGraph cognition state 正式贯通到 Go plan/session/trace summary、presentation、startup diagnostics、smart 结果页与 preview mock，同时补齐 sidecar fallback/degraded 相关测试。
- 最终目标：让 LangGraph 作为非阻塞认知层稳定接入，sidecar 不可用时自动回退 deterministic planner；presentation 与 smart 页统一消费短解释；trace 中同时呈现 Go 决策轨迹和 LangGraph cognition 摘要。
- 当前采用的方法：
  - Go 继续作为唯一执行真相源，负责 deterministic planner、validation、execution、rollback、session persistence 与 trace 汇总；LangGraph 只提供 candidate preference 与短解释。
  - `AgentPlan`、`AgentSession.Context`、`TraceSummary` 和 `agent.explanation` 统一携带结构化 `cognition`/`cognition_state`，前后端都通过同一份状态语义展示 engaged/degraded/fallback/disabled/unavailable。
  - sidecar 健康、planner fallback、explain 降级与 startup checks 都统一映射为短结构化状态和 fallback reason code，不把原始 Python 异常直接暴露给 UI。
- 相关模块/文件：
  - `appshell/backend/internal/agent/cognition.go`
  - `appshell/backend/internal/agent/actions.go`
  - `appshell/backend/internal/agent/types.go`
  - `appshell/backend/internal/agent/helpers.go`
  - `appshell/backend/internal/agent/mock_planner.go`
  - `appshell/backend/internal/agent/langgraph_planner.go`
  - `appshell/backend/internal/agent/planning_flow.go`
  - `appshell/backend/internal/agent/runtime_runner.go`
  - `appshell/backend/internal/agent/cognition_test.go`
  - `appshell/backend/internal/agent/langgraph_planner_test.go`
  - `appshell/backend/internal/agent/runtime_runner_test.go`
  - `appshell/backend/internal/presentation/builder_agent.go`
  - `appshell/backend/internal/presentation/builder_test.go`
  - `appshell/backend/cmd/wails/app.go`
  - `appshell/backend/cmd/wails/app_test.go`
  - `appshell/backend/cmd/wails/startup_checks.go`
  - `appshell/backend/cmd/wails/startup_checks_test.go`
  - `appshell/frontend/src/main.js`
  - `tests/langgraph_sidecar/test_graph.py`
  - `tests/langgraph_sidecar/test_server.py`
  - `MEMO.md`
- 已完成的步骤 / 已解决的问题 / 新增功能：
  - 后端新增统一 `AgentCognitionState / CognitionTraceSummary`，并把 `cognition` 正式加入 `AgentPlan`、`TraceSummary`、`agent.explanation` 与 `AgentSession.Context["cognition_state"]`。
  - `LangGraphPlanner.BuildPlan(...)` 现在无论成功、degraded 还是 fallback 都会回填 cognition state，并统一给出 `disabled / script_missing / startup_failed / port_occupied / healthcheck_failed / planner_mode_fallback / plan_request_failed / invalid_candidate / explain_request_failed` 等 reason code。
  - planning flow 固定写入 `cognition_trace`，`SummarizeTraceEvents(...)` 统一汇总 cognition 摘要，Wails `GetAgentSession` 与 runtime 返回复用同一份 trace summary 逻辑。
  - presentation builder 已优先消费 `agent.explanation`，并把 cognition status、summary 与 fallback reason 接到 highlights、strategy section 和 trace timeline。
  - startup diagnostics 现在会稳定区分 `LLM ready`、`fallback active`、`disabled` 等状态，并把 provider / planner_mode / llm_mode / cognition_status / fallback_reason_code 暴露给前端。
  - smart 结果页与 reasoning/trace summary 现在会展示 `LangGraph: provider/status`、cognition summary 和 fallback reason；startup 卡也能直接看出当前是 langgraph engaged 还是 deterministic fallback。
  - preview mock 已升级为带 cognition 的两段式结果源：mock `agent.explanation`、`plan.cognition`、`session.context.cognition_state`、`trace_summary.cognition` 与 `cognition_trace` 都已补齐，方便离线演示 Phase E。
  - Go 测试新增并通过：
    - planner engaged / health fallback / planner_mode fallback / invalid candidate / plan request failure / explain degraded 的 cognition 断言
    - runtime plan/approval 路径中的 `cognition_state`、`cognition_trace` 与 trace summary 断言
    - `SummarizeTraceEvents(...)` 与 `buildAgentExplanationPayload(...)` 的专门 cognition 单测
    - presentation / startup checks / Wails session 读取中的 cognition 汇总断言
  - Python sidecar 测试新增并通过：
    - `/health` 在 LLM 已配置时返回稳定的 `planner_mode=llm` / `llm_mode=configured`
    - 在本机未安装 `langgraph` 依赖时，相关 graph/server 测试会自动 skip，而不是直接 collection fail
- 已完成验证：
  - `go test ./...`（`appshell/backend`）通过
  - `node --check appshell/frontend/src/main.js` 通过
  - `python -m pytest tests/langgraph_sidecar -q` 通过，结果为 `7 passed, 2 skipped`
- 当前问题 / 待处理事项：
  - 仍需做一轮真实 Wails 桌面端手工联调，重点确认 smart 结果页、startup diagnostics 与 preview mock 外的真实 binding 数据在 UI 上的最终文案和排版。
  - `appshell/frontend/src/main.js` 内部仍存在一些历史乱码文案与 preview mock 旧结构，虽然当前语法与行为已通过检查，但后续值得再做一次纯净化整理。
  - LangGraph sidecar 的 Python 测试现在会在缺少 `langgraph` 依赖时自动 skip；若后续要把这部分纳入更严格 CI，需要在 CI 环境显式安装对应依赖。

## Update 2026-03-17 20:34:49

- 改动日期：2026-03-17 20:34:49
- 改动内容简述：新增 LANGGRAPH_DEEPENING_EXECUTION_PLAN.md，把 LangGraph 下一阶段升级路线收敛为可直接接力执行的工程文档，明确子图拆分、graph 级中断与恢复、更完整的认知轨迹、角色化认知协作四条升级主线，并为每个阶段补充范围、涉及模块、实施步骤、非目标与验收标准。
- 相关模块/文件：
  - LANGGRAPH_DEEPENING_EXECUTION_PLAN.md
  - MEMO.md
- 已解决的问题/新增功能：
  - 新增独立的 LangGraph 深化执行文档，可在新窗口中直接作为继续推进的主说明。
  - 明确当前基线：Go control plane 仍是执行与回退真相来源，LangGraph 继续定位为 cognition runtime。
  - 明确后续阶段顺序：先做契约冻结与子图拆分，再进入 interrupt/resume、trace 映射、角色协作与产品化。
  - 为每个阶段写入了清晰验收标准，避免未来再次回到“要不要做、先做什么、做到什么算完成”的讨论。
- 待处理事项：
  - Phase 0：冻结 /health、/v1/plan、/v1/explain 协议与 cognition trace 最小事件模型。
  - Phase 1：将当前线性 intent -> strategy -> explain 重构为可复用 subgraph，且不改变现有外部行为。
  - Phase 2：为审批引入 graph 级 interrupt/resume 与可重启恢复的 checkpoint 存储。
  - Phase 3：把 LangGraph 节点级 trace 映射进 Go agent_trace 与 presentation。
  - Phase 4：基于结构化状态引入条件路由的 role-based cognition collaboration，而不是自由聊天式多 agent。
  - Phase 5：完善 startup checks、presentation、handoff 规范，并持续更新本备忘录与新路线文档。
- 最终目标：让 LangGraph 在本项目中从“线性 cognition overlay”升级为“可拆分、可中断、可恢复、可追踪、可条件协作”的 cognition runtime，同时不破坏 Go control plane、deterministic tool layer 与 fallback safety。
- 我们正在采取的方法：采用分阶段推进策略，先固化契约与子图边界，再逐步引入 checkpoint、interrupt/resume、trace 融合与角色化协作，确保每一阶段都可单独测试、单独验收、单独回退。
- 我们目前已完成的步骤：已完成 LangGraph sidecar 的基础接入、/health、/v1/plan、/v1/explain、Go 侧 planner overlay/fallback、startup checks、session/trace 基础持久化，以及本次新增的深化执行路线文档。
- 我们当前正在努力解决的问题：当前 LangGraph 仍是单线节点图，缺少 subgraph、graph-native interrupt/resume、节点级 cognition trace 与真正的条件化角色协作；这些能力已经在新路线文档中拆解为后续可执行 phase。

## Update 2026-04-06 17:08:39

- 改动日期：2026-04-06 17:08:39
- 改动内容简述：补充一套可复现的后端性能实验入口，新增 `appshell/backend/cmd/bench` 统一执行 Go 调度层 synthetic benchmark、审批暂停/恢复 synthetic benchmark 和真实 `agent.session.plan` 端到端 benchmark，并完成一轮实测，输出可直接复用到简历、答辩和后续优化工作的量化数据。
- 最终目标：在不改变当前 Go control plane、Python tool layer 和 LangGraph fallback safety 边界的前提下，为“高并发调度”“人工审批恢复”“计划链路稳定性”提供可重复、可落盘、可追溯的工程量化指标。
- 当前采用的方法：
  - 在 `appshell/backend/cmd/bench/main.go` 中统一封装三类实验：Synthetic Scheduler、Synthetic Approval Resume、E2E Agent Plan。
  - 使用项目内 `.venv-win` Python 解释器、`data/raw/simple_obvious_anomaly.csv` 和 `outputs/results/wails_mvp` 作为稳定基线，先以 `APPSHELL_LANGGRAPH_ENABLED=0` 固化 deterministic baseline，避免把外部 provider 波动混入当前数字。
  - 将实验报告写入 `outputs/results/backend_benchmark_latest.json`，并在 `appshell/backend/README.md` 中补充可直接执行的 benchmark 命令。
- 相关模块/文件：
  - `appshell/backend/cmd/bench/main.go`
  - `appshell/backend/README.md`
  - `outputs/results/backend_benchmark_latest.json`
  - `MEMO.md`
- 已完成的步骤 / 已解决的问题 / 新增功能：
  - 新增统一 benchmark CLI：`go run ./cmd/bench -scenario all -python-bin ... -plan-csv ... -plan-model-dir ...`
  - Synthetic Scheduler 实测完成：在 60 个、单任务 40ms 的 synthetic 任务下，worker 从 1 提升到 6 时，吞吐从 `24.76 tasks/s` 提升到 `148.39 tasks/s`，`P95` 端到端时延从 `2303.93ms` 降到 `404.19ms`，相对单 worker 达到 `5.99x` 加速。
  - Synthetic Approval Resume 实测完成：30 轮审批暂停/恢复全部成功，成功率 `100%`；平均暂停延迟 `197.46ms`，平均恢复延迟 `120.02ms`，平均往返延迟 `317.48ms`，trace 持久化命中率 `100%`。
  - E2E Agent Plan 实测完成：对 `simple_obvious_anomaly.csv` 在 1 轮 warmup 后连续执行 3 轮 `agent.session.plan`，平均总耗时 `4966.67ms`，`P95` 总耗时 `5140.20ms`，队列等待基本可忽略（平均 `0.02ms`）；3 轮均稳定选出 `hybrid` 方案。
  - 本轮 E2E benchmark 已给出瓶颈定位：平均最慢阶段为 `agent_retrieve`，平均 `3864ms`；其后为 `validate_input`（平均 `2149.67ms`）与 `agent_profile/agent_scan`（各 `1012.67ms`）。
  - 回归验证完成：`go test ./...`（`appshell/backend`）通过。
- 当前问题 / 待处理事项：
  - 当前端到端 plan 路径的主瓶颈仍在 `agent_retrieve`，后续如需进一步压缩 plan 耗时，可优先评估 Gower preview 复用、结果缓存或 Python 进程常驻化方案。
  - 当前仓库新增的 benchmark 覆盖的是 Go 调度、审批恢复和现有 plan 链路；如果后续需要把 `gRPC / Redis / PostgreSQL` 写成带量化支撑的简历条目，仍需在对应实现仓库或服务层补独立压测。
  - 当前 benchmark 默认固定在 LangGraph disabled baseline；若后续接入真实 provider 并希望量化 cognition runtime 的真实耗时，需要在配置好 provider 后再次执行 `cmd/bench` 做一轮 enabled 对照实验。

## Update 2026-04-06 17:30:41

- 改动日期：2026-04-06 17:30:41
- 改动内容简述：围绕 `agent_retrieve` 做一轮可复现的优化实验，新增 `sequential / parallel` 双模式预览执行路径与 A/B benchmark 入口；同时修复并发实验下 SQLite trace 持久化偶发 `SQLITE_BUSY` 的稳定性问题，并输出可直接用于答辩或简历的对比数据。
- 最终目标：在不改变当前 Go control plane、Python engine 和 deterministic planner 安全边界的前提下，缩短 `agent.session.plan` 中 `agent_retrieve` 阶段的真实墙钟耗时，并保留方案选择的一致性与 trace 可追溯性。
- 当前采用的方法：
  - 在 planning flow 中为 `agent_retrieve` 引入 `agent_retrieve_mode`，支持 `sequential` 与 `parallel` 两种 preview 执行模式，并通过同一条 benchmark 命令入口做 A/B 对照。
  - 将 rule preview 与 Gower preview 的真实工具调用收敛到 `runPreviewTools(...)`，在并行模式下并发执行，但保持 trace 写入仍由主流程顺序落盘。
  - 为 `SQLiteStore` 增加单连接与写入串行化约束，降低长链路 benchmark 中 trace/session 持久化受 SQLite 锁竞争影响的概率。
  - 使用 1 轮 warmup + 5 轮 measured、同一份 CSV / model_dir、LangGraph disabled baseline，对 `sequential` 和 `parallel` 做同口径压测。
- 相关模块/文件：
  - `appshell/backend/internal/agent/planning_flow.go`
  - `appshell/backend/internal/agent/retrieve_mode.go`
  - `appshell/backend/internal/agent/retrieve_preview.go`
  - `appshell/backend/internal/agent/retrieve_preview_test.go`
  - `appshell/backend/internal/agent/sqlite_store.go`
  - `appshell/backend/internal/agent/sqlite_store_test.go`
  - `appshell/backend/cmd/bench/main.go`
  - `appshell/backend/README.md`
  - `outputs/results/backend_benchmark_retrieve_sequential.json`
  - `outputs/results/backend_benchmark_retrieve_parallel.json`
  - `outputs/results/backend_benchmark_retrieve_compare.json`
  - `MEMO.md`
- 已完成的步骤 / 已解决的问题 / 新增功能：
  - `agent.session.plan` 现已支持通过 payload / env 控制 `agent_retrieve_mode`，并把该模式写入 session context 与 stage metadata，便于 benchmark 与 trace 复盘。
  - `agent_retrieve` 的 rule preview 与 Gower preview 已支持并行实验路径；在真实 benchmark 中，`parallel` 模式可稳定完成 5/5 轮端到端计划任务。
  - `cmd/bench` 新增 `-agent-retrieve-mode` 参数，支持直接生成 `sequential / parallel` 两份独立报告，并在 README 中补充了 A/B 运行示例。
  - `SQLiteStore` 已增加单连接和写入锁；新增并发 trace 写入测试，连续多次运行均通过，避免本轮优化被 SQLite trace 锁竞争放大为假性失败。
  - 本轮 A/B 实测结果如下：
    - `sequential`：平均总耗时 `4317.0ms`，`P95` 总耗时 `4524.2ms`，`agent_retrieve` 平均 `3344.2ms`。
    - `parallel`：平均总耗时 `3817.6ms`，`P95` 总耗时 `4103.0ms`，`agent_retrieve` 平均 `2823.4ms`。
    - 相比 `sequential`，`parallel` 将总耗时降低 `499.4ms`（`11.57%`），将 `P95` 总耗时降低 `421.2ms`（`9.31%`），将 `agent_retrieve` 阶段耗时降低 `520.8ms`（`15.57%`）。
    - 两组实验均连续 `5/5` 轮稳定选出 `hybrid` 方案，说明本轮优化未引入 candidate selection 漂移。
- 已完成验证：
  - `go test ./internal/agent -run "Test(SQLiteStore|RetrievePreview)"` 通过
  - `go test ./internal/agent -run "TestSQLiteStoreSaveTraceEventConcurrentWriters" -count 10` 通过
  - `go run ./cmd/bench ... -agent-retrieve-mode sequential` 通过，并输出 `outputs/results/backend_benchmark_retrieve_sequential.json`
  - `go run ./cmd/bench ... -agent-retrieve-mode parallel` 通过，并输出 `outputs/results/backend_benchmark_retrieve_parallel.json`
- 当前问题 / 待处理事项：
  - 本轮并行优化已把 `agent_retrieve` 从 `3344.2ms` 压到 `2823.4ms`，但它仍是当前 plan 链路最慢阶段，后续仍可继续评估 Gower 检索复用、Redis/内存缓存和 Python worker 常驻化。
  - `go test ./...` 在当前机器上仍会被系统默认 `python` 缺失拖挂若干历史测试（`exit status 9009`）；本轮相关定向测试已通过，但若要恢复全量回归，需要补齐测试环境中的默认 Python 可执行入口。
  - 本轮 `parallel` 模式已经可稳定跑 benchmark，但如果后续要把该模式作为默认生产路径，仍建议在真实 Wails 桌面链路上再做一轮 session/trace/UI 联调。
- 我们目前已完成的步骤：已完成 `agent_retrieve` 的双模式执行接入、并发 preview 回归测试、SQLite trace 写入稳定性修复、benchmark CLI A/B 开关、README 命令补充，以及一轮真实端到端 A/B benchmark。
- 我们当前正在努力解决的问题：虽然并行 preview 已经带来双位数的总耗时下降，但 `agent_retrieve` 仍是首要瓶颈；下一阶段需要继续判断瓶颈主要来自 Gower 检索本身、Python 进程生命周期，还是可复用中间结果尚未缓存。

## Update 2026-04-11 18:16:19

- 改动日期：2026-04-11 18:16:19
- 改动内容简述：重装系统后对当前仓库的 Git 环境做了一轮恢复与连通性验证，补齐 Git for Windows、本机全局 Git 配置、仓库 `safe.directory` 和 GitHub 凭据链路，并确认该仓库重新具备稳定的远程读取能力与可用的推送认证环境。
- 最终目标：让这台重装后的 Windows 机器重新回到“能正常进入仓库、能安全识别仓库所有权、能从 GitHub 拉取、能在需要时继续向原仓库推送”的可开发状态，而不是只保留本地 `.git` 目录。
- 当前采用的方法：
  - 先读取仓库现有 `MEMO.md` 与 `.git/config`，确认项目历史远程地址和当前工作区状态，再决定恢复步骤。
  - 使用 `winget` 安装官方 `Git for Windows`，避免手工拼装环境变量或使用来源不明的第三方包。
  - 基于仓库历史提交记录恢复全局 `user.name` / `user.email`，并把当前仓库加入 `safe.directory`，修复系统重装后 SID 变化导致的 `dubious ownership` 问题。
  - 通过 `fetch`、`pull --ff-only`、`push --dry-run`、GitHub 凭据读取和远程引用读取，多角度验证远程连通性与认证状态，而不对远端分支做真实改写。
- 相关模块/文件：
  - `MEMO.md`
  - `~/.gitconfig`
- 已完成的步骤 / 已解决的问题 / 新增功能：
  - 确认仓库原有远程地址仍然存在：`origin` 继续指向 `https://github.com/HITGerhman/Anomaly-Detection-and-Repair-for-Mixed-Data-Type-Inputs.git`。
  - 确认本机网络可访问 GitHub，`https://github.com` 与该仓库页面都返回正常状态。
  - 使用 `winget install Git.Git` 成功安装 `Git for Windows 2.53.0.2`。
  - 确认系统级 `PATH` 已写入 `C:\Program Files\Git\cmd`；当前旧会话未自动刷新，但新开的终端应可直接使用 `git`。
  - 修复仓库安全目录问题：已将 `D:/code/pythoncode/Anomaly Detection and Repair for Mixed Data Type Inputs` 写入全局 `safe.directory`，现在 Git 不再因系统重装后的所有者 SID 变化拒绝访问仓库。
  - 恢复全局 Git 身份：已写入 `user.name=HITGerhman`、`user.email=2727190275@qq.com`，与该仓库最近提交历史保持一致。
  - 确认 Git Credential Manager 已随 Git for Windows 安装并启用，`credential.helper=manager` 已写入全局配置。
  - 确认本机当前存在 GitHub HTTPS 凭据，读取结果显示 `username=HITGerhman` 且已有可用密码/令牌条目。
  - 实际验证远程读取链路：`git fetch origin --verbose` 成功，`git pull --ff-only` 返回 `Already up to date.`，`git ls-remote --heads origin` 能正常读取远端 `main`。
  - 实际验证推送认证链路：`git push --dry-run origin HEAD:main` 返回 `Everything up-to-date`，说明当前认证环境未阻塞针对 `origin/main` 的推送协商。
- 当前问题 / 待处理事项：
  - 当前 Codex 运行会话仍继承了安装 Git 前的旧 `PATH`，所以本会话里直接键入 `git` 可能仍不可用；新开的 PowerShell / CMD / VS Code 终端应恢复正常，如仍异常可重开终端或重新登录一次系统。
- 本轮为了避免污染远端仓库，没有做“创建临时分支再删除”的真实写入探针；当前对 push 的判断基于已存在凭据和 `git push --dry-run` 成功。
- 若后续需要在这台机器上新建提交，建议额外执行一次 `git config --global --list` 和 `git status` 自检，确保你的日常终端环境已经刷新到新的 `PATH`。
- 我们目前已完成的步骤：已确认远程仓库地址未丢失、网络连通正常、Git for Windows 安装完成、仓库安全目录问题已修复、全局提交身份已恢复、GitHub 凭据链路已存在，并完成了当前仓库的 fetch / pull / push dry-run 验证。
- 我们当前正在努力解决的问题：当前剩下的唯一小尾巴是“旧终端会话还没刷新到新的 `PATH`”；这不影响新开的终端和后续开发，但如果用户继续沿用老窗口，需要重开终端后再直接使用 `git` 命令。

## Update 2026-04-11 18:27:16

- 改动日期：2026-04-11 18:27:16
- 改动内容简述：对当前工作区的大批量本地改动完成了一轮提交前整理，确认这是一笔覆盖 agent runtime、LangGraph sidecar、presentation、Wails 前端、环境脚本、benchmark 和路线文档的阶段性快照；基于风险控制改为新建分支 `feat/agent-runtime-langgraph-bench` 承载本次提交，而不是直接推送到 `main`。
- 最终目标：在不丢失当前完整阶段成果的前提下，把本地积累已久的大批量改动安全推送到 GitHub，并保留后续继续 review、补 Node 环境验证、再决定是否合并进 `main` 的缓冲空间。
- 当前采用的方法：
  - 先用 `git status --short --branch`、`git diff --stat`、`git ls-files --others --exclude-standard` 汇总修改文件、未跟踪文件和整体改动规模。
  - 先做最小必要清理，只修复 `MEMO.md` 与 `requirements.txt` 末尾空行，让 `git diff --check` 不再报真正的文本格式问题。
  - 在不改业务代码的前提下做关键校验：Python engine 回归、LangGraph sidecar 回归、Go 关键包回归；把环境缺失导致的失败与代码真实失败分开判断。
  - 因为当前改动量较大且前端 `node` 环境尚未恢复，所以选择新分支隔离风险，再进行 commit / push。
- 相关模块/文件：
  - `MEMO.md`
  - `requirements.txt`
  - `.gitignore`
  - `README.md`
  - `app.py`
  - `appshell/`
  - `src/`
  - `tests/`
  - `scripts/`
  - `requirements.lock.txt`
  - `ENVIRONMENT.md`
  - `LANGGRAPH_DEEPENING_EXECUTION_PLAN.md`
  - `LANGGRAPH_UPGRADE_ROADMAP.md`
  - `MULTI_AGENT_BLUEPRINT.md`
  - `PRESENTATION_CATALOG.md`
  - `TOOL_LAYER_FOUNDATION.md`
- 已完成的步骤 / 已解决的问题 / 新增功能：
  - 已确认当前工作区不是零散修补，而是一整批阶段性成果：既包含已有文件的持续演进，也包含大量新增模块、测试、脚本和设计文档。
  - 已完成本地范围梳理：跟踪到的已修改文件 19 个，另有大批新增文件与目录尚未纳入 Git。
  - 已修复提交前文本检查中唯一明确的格式问题：`MEMO.md` 与 `requirements.txt` 的 EOF 多余空行。
  - `tests/python_engine -q` 已通过，结果为 `29 passed`。
  - `tests/langgraph_sidecar -q` 已通过，结果为 `12 passed`。
  - `go test ./...` 在当前机器初次执行时失败，根因不是这批代码，而是系统默认 `python` 命令缺失，导致若干 Go 测试中的 Python 子进程启动返回 `exit status 9009`。
  - 在临时将项目 `.venv-win\\Scripts` 注入 `PATH` 后，`go test ./internal/engine ./internal/agent ./cmd/wails` 已通过，说明本轮核心 Go 改动本身具备可提交性。
  - 当前系统仍未恢复 `node`，因此 `node --check appshell/frontend/src/main.js` 无法执行；这属于机器环境缺口，而不是已经定位出的前端语法错误。
  - 已新建并切换到提交分支：`feat/agent-runtime-langgraph-bench`。
- 当前问题 / 待处理事项：
  - 仍需完成本轮 Git 操作的最后两步：把当前分支的全部目标改动暂存并创建 commit，然后推送到 `origin/feat/agent-runtime-langgraph-bench`。
  - 当前机器尚未安装或恢复 `node`，所以这次提交前无法完成前端语法层面的本机验证；后续如果要收口到更严格的发布质量，建议补齐 Node 环境后再执行前端检查。
  - 若后续希望 `go test ./...` 在任意新终端直接通过，需要把项目 `.venv-win\\Scripts` 或可用的系统 `python` 命令恢复到常规开发环境，而不只是在单次命令里临时注入。
- 我们目前已完成的步骤：已完成改动范围梳理、EOF 小清理、关键 Python 回归、关键 Go 回归、风险判断与分支策略选择，并已把工作区切换到 `feat/agent-runtime-langgraph-bench`。
- 我们当前正在努力解决的问题：接下来只剩提交与推送收口，以及系统级 `node/python PATH` 环境还不够干净这两个收尾问题；前者本轮会完成，后者会作为环境待办继续保留。

## Update 2026-04-11 18:29:03

- 改动日期：2026-04-11 18:29:03
- 改动内容简述：完成当前阶段性成果的 Git 收口，已在分支 `feat/agent-runtime-langgraph-bench` 上创建正式 commit 并成功推送到 GitHub，远端跟踪关系和 PR 入口已建立。
- 最终目标：把本地长时间积累的大批量阶段成果先安全落到远端分支，确保代码、文档、测试和环境脚本都有可恢复的 GitHub 备份，再视 review 结果决定是否合并进 `main`。
- 当前采用的方法：
  - 先把全部目标文件统一加入暂存区，确认 staged 统计与预期一致。
  - 使用单个阶段性 commit 作为当前里程碑快照，避免在尚未补 Node 环境前拆出多个语义不完整的小提交。
  - 通过 `git push -u origin feat/agent-runtime-langgraph-bench` 一次性创建远端分支并绑定上游，方便后续继续增量提交。
- 相关模块/文件：
  - `MEMO.md`
  - 当前分支全部已纳入 Git 的阶段性改动文件
- 已完成的步骤 / 已解决的问题 / 新增功能：
  - 已将本轮目标改动全部暂存，并确认 staged 范围覆盖 91 个文件。
  - 已创建阶段性 commit：`bd6d8a056b868f9ff1dcf030c2ca91c98980448f`
  - commit message：`feat: land agent runtime langgraph and benchmark tooling`
  - 已成功推送到远端新分支：`origin/feat/agent-runtime-langgraph-bench`
  - 已建立分支上游跟踪关系，后续在该分支上可直接继续 `git push` / `git pull`
  - GitHub 已生成可直接发起合并请求的入口：
    - `https://github.com/HITGerhman/Anomaly-Detection-and-Repair-for-Mixed-Data-Type-Inputs/pull/new/feat/agent-runtime-langgraph-bench`
- 当前问题 / 待处理事项：
  - 当前 `MEMO.md` 这条“已推送”记录还需要作为一个小补充 commit 再推送一次，才能让仓库内备忘与远端状态完全一致。
  - 远端分支已经存在，但是否立即创建 PR、是否直接合并到 `main`，仍建议在补齐 Node 环境或完成一轮前端实机验证后再决定。
  - 当前机器的 `node` 缺失、系统级 `python PATH` 不够干净的问题依然存在，后续如果继续在这台机器上开发，值得单独做一次环境收口。
- 我们目前已完成的步骤：已完成分支创建、全部改动的 commit、远端推送、上游绑定和 PR 入口生成。
- 我们当前正在努力解决的问题：此刻只剩最后一个很小的收尾动作，即把这条推送记录本身也提交并同步到同一远端分支；功能层面的主成果已经安全备份到 GitHub。

## Update 2026-05-03 18:29:16

- 改动日期：2026-05-03 18:29:16 +08:00
- 改动内容简述：执行毕业设计路线图 M0“项目基线确认”，新增项目基线说明文档，并在路线图中更新 M0 状态、说明和验证命令；本次只做基线确认与文档记录，不执行 M1 及之后任务，不重构主架构，不引入新依赖。
- 最终目标：在继续推进毕业设计收尾前，先形成当前项目真实状态快照，让后续开发者能够知道项目如何启动、如何验证、哪些入口可用、哪些环境问题和核心模块风险需要谨慎处理。
- 当前采用的方法：
  - 先读取 `GRADUATION_PROJECT_ROADMAP.md` 和当前 `MEMO.md`，确认 M0 的目标和边界。
  - 用现有 `.venv-win`、Python engine、Go backend 和 Node/npm 命令做基线验证，只记录真实结果，不修复环境、不安装依赖。
  - 将验证结果沉淀到 `PROJECT_BASELINE.md`，并让路线图只保留总纲级状态和命令摘要。
- 相关模块/文件：
  - `PROJECT_BASELINE.md`
  - `GRADUATION_PROJECT_ROADMAP.md`
  - `MEMO.md`
- 已完成的步骤 / 已解决的问题 / 新增功能：
  - 新增 `PROJECT_BASELINE.md`，记录当前主入口、环境状态、验证命令、已知问题和高风险模块。
  - 确认当前主入口包括：`app.py` Streamlit 路径、Python engine JSON 协议入口、Go backend demo、Wails shell backend、frontend 静态预览。
  - 确认 `.venv-win` Python 可用：`Python 3.11.7`。
  - Python engine 测试收集通过：`21 tests collected in 1.68s`。
  - Python engine 测试执行通过：`21 passed in 33.86s`。
  - Engine health 验证通过，返回 `status=ok`，并确认支持 `health/train/repair/scan_file/repair_batch/rollback_repair_batch`。
  - Go 关键包在默认环境下暴露真实问题：`appshell/backend/internal/engine` 因 Python 子进程入口返回 `exit status 9009` 失败；`internal/task` 和 `cmd/wails` 通过。
  - 临时将 `.\.venv-win\Scripts` 加入 `PATH` 后，`go test ./internal/engine ./internal/task ./cmd/wails` 通过，说明 Go 侧失败主要来自默认 Python 入口环境，而不是本轮代码改动。
  - 确认 Node/npm 当前不可作为可靠前端基线：`node --version` 失败为 `Access is denied`，`npm --version` 失败为命令不存在。
  - 已在 `GRADUATION_PROJECT_ROADMAP.md` 中将 M0 状态更新为 `DONE`，并补充完成说明、验证命令和结果摘要；M1-M6 仍保持 `TODO`。
- 当前问题 / 待处理事项：
  - 默认 shell 的 Python 入口仍不适合作为 Go engine runner 测试依赖；后续如需让 Go 测试在任意终端稳定通过，需要把 `.venv-win\Scripts` 或可用系统 Python 放入常规 `PATH`。
  - Node/npm 当前不可用，前端构建、lint 或打包验证仍需后续单独恢复环境后再做。
  - 根目录 `README.md` 与 AppShell 当前能力仍存在文档漂移，后续需要在独立任务中同步。
  - Windows clean-machine 安装包验证仍未完成，不属于本次 M0 的已通过基线。
  - 当前已有未跟踪项 `out/figma-verify/` 与 `scripts/langgraph.local.ps1` 未处理，保持原样。
- 我们目前已完成的步骤：已完成 M0 项目基线确认、基线文档新增、路线图 M0 状态更新，以及本次改动记录。
- 我们当前正在努力解决的问题：把项目从“能跑的工程原型”收束为“可证明、可复现、可测试、可演示、可写论文”的毕业设计交付状态；本次已先完成后续工作的真实基线锚点。

## Update 2026-05-03 18:47:40

- 改动日期：2026-05-03 18:47:40 +08:00
- 改动内容简述：执行毕业设计路线图 M1“实验数据与异常注入体系”，新增可复现实验数据生成脚本、M1 专项测试和 `data/experiments/m1_stroke/` 数据产物，并在路线图中更新 M1 状态、说明和验证命令；本次不执行 M2 及之后任务，不修改核心算法、Python engine 协议、Go 后端或 Wails 前端，不引入新依赖。
- 最终目标：为后续 M2 检测效果评估和 M3 修复效果评估提供可信的 clean/corrupted/ground truth 基础数据，使毕业设计证明链条从“系统能运行”推进到“结果可复现、可量化”。
- 当前采用的方法：
  - 使用 M0 已确认可用的 `.venv-win` Python 环境执行生成与验证。
  - 选择混合类型且体量适中的 `data/raw/healthcare-dataset-stroke-data.csv` 作为 M1 主数据源。
  - 先生成保守 clean subset，再用固定随机种子 `20260503` 注入当前系统可扫描和可解释的五类异常。
  - 将注入事实写入 `ground_truth.csv`，将统计信息写入 `injection_summary.json`，不在 M1 阶段计算检测或修复指标。
- 相关模块/文件：
  - `.gitignore`
  - `scripts/generate_m1_experiment_data.py`
  - `tests/python_engine/test_m1_experiment_data.py`
  - `data/experiments/m1_stroke/clean.csv`
  - `data/experiments/m1_stroke/corrupted.csv`
  - `data/experiments/m1_stroke/ground_truth.csv`
  - `data/experiments/m1_stroke/injection_summary.json`
  - `data/experiments/m1_stroke/README.md`
  - `GRADUATION_PROJECT_ROADMAP.md`
  - `MEMO.md`
- 已完成的步骤 / 已解决的问题 / 新增功能：
  - 新增 `scripts/generate_m1_experiment_data.py`，支持默认从 stroke 原始数据生成 M1 实验数据，也支持自定义 `--source-csv`、`--output-dir` 和 `--seed`。
  - clean subset 已删除缺失行，移除天然极稀有类别，并过滤 `age`、`avg_glucose_level`、`bmi` 的明显数值尾部，以降低原始噪声对 ground truth 的污染。
  - 为实验数据新增 `row_id`、`source_row_id`、`record_start_day`、`record_end_day`，其中正常数据满足 `record_start_day <= record_end_day`。
  - 已注入五类 M1 范围异常：`missing_values=30`、`numeric_outlier=24`、`rare_category=18`、`duplicate_record=12`、`cross_column_consistency=16`。
  - 已生成 `clean.csv`：4228 行、16 列。
  - 已生成 `corrupted.csv`：4240 行、16 列。
  - 已生成 `ground_truth.csv`：100 条注入记录，字段固定为 `injection_id/anomaly_type/expected_issue_type/row_id/source_row_id/row_index/column/original_value/corrupted_value/repairable/notes`。
  - 已生成 `injection_summary.json` 和数据目录 `README.md`，记录种子、行列数、注入数量和 M1/M2/M3 边界。
  - 新增 `tests/python_engine/test_m1_experiment_data.py`，覆盖文件存在性、数量关系、注入统计一致性、clean/corrupted 基本约束和重复运行稳定性。
  - 已在 `.gitignore` 中为 M1 的三个小型 CSV 数据资产添加例外，确保后续提交时不会被全局 `*.csv` 规则静默忽略。
  - 已在 `GRADUATION_PROJECT_ROADMAP.md` 中将 M1 状态更新为 `DONE`，并补充完成说明、验证命令和结果摘要；M2-M6 仍保持 `TODO`。
- 验证结果：
  - `.\.venv-win\Scripts\python.exe scripts\generate_m1_experiment_data.py --output-dir data\experiments\m1_stroke --seed 20260503` 执行成功。
  - `.\.venv-win\Scripts\python.exe -m pytest tests/python_engine/test_m1_experiment_data.py -q` 通过：`2 passed in 2.70s`。
  - `.\.venv-win\Scripts\python.exe -m pytest tests/python_engine -q` 通过：`23 passed in 25.58s`。
  - scan smoke test 通过，engine 返回 `status=ok`，读取 `corrupted.csv` 的 4240 行、16 列，汇总 `issue_count=19`，包含缺失值、数值离群、稀有类别、重复记录和跨列一致性问题。
- 当前问题 / 待处理事项：
  - M1 只完成实验数据与 ground truth 构造；检测指标、混淆统计和评估报告留给 M2。
  - M1 不计算修复指标；修复效果评估留给 M3。
  - 本次未处理默认 Python PATH、Node/npm 和前端构建环境问题，这些仍沿用 M0 记录的基线结论。
  - 当前已有未跟踪项 `out/figma-verify/` 与 `scripts/langgraph.local.ps1` 未处理，保持原样。
- 我们目前已完成的步骤：已完成 M0 项目基线确认，以及 M1 可复现实验数据、异常注入记录、统计摘要、专项测试和路线图状态更新。
- 我们当前正在努力解决的问题：继续把项目收束为可证明的毕业设计交付物；下一步只有在明确执行 M2 时，才应基于本次 `data/experiments/m1_stroke/` 计算检测效果指标。

## Update 2026-05-03 19:02:57

- 改动日期：2026-05-03 19:02:57 +08:00
- 改动内容简述：执行毕业设计路线图 M2“异常检测效果评估”，新增检测评估脚本、M2 专项测试和 `data/experiments/m2_stroke_detection/` 评估产物，并在路线图中更新 M2 状态、说明和验证命令；本次不执行 M3 及之后任务，不修改核心算法、Python engine 协议、Go 后端或 Wails 前端，不引入新依赖。
- 最终目标：基于 M1 的可控注入数据和 ground truth，量化当前扫描器发现真实异常的能力，形成可写入论文和答辩材料的检测指标与问题分析。
- 当前采用的方法：
  - 使用 `.venv-win` Python 环境运行 M2 评估。
  - 以 `data/experiments/m1_stroke/ground_truth.csv` 作为唯一真值来源。
  - 评估脚本调用现有 `engine_core` 内部检测函数取得完整 mask，但不改 `scan_file` 对外协议。
  - 评分口径为：缺失值、数值离群、稀有类别按 `anomaly_type + row_index + column` 精确匹配；跨列一致性按 `anomaly_type + row_index` 匹配；重复记录按 `anomaly_type + source_row_id` 组匹配。
  - M2 关闭 `time_series_shift` 评分，因为 M1 未注入该类异常。
- 相关模块/文件：
  - `scripts/evaluate_m2_detection.py`
  - `tests/python_engine/test_m2_detection_evaluation.py`
  - `data/experiments/m2_stroke_detection/detection_metrics.json`
  - `data/experiments/m2_stroke_detection/detection_matches.json`
  - `data/experiments/m2_stroke_detection/README.md`
  - `GRADUATION_PROJECT_ROADMAP.md`
  - `MEMO.md`
- 已完成的步骤 / 已解决的问题 / 新增功能：
  - 新增 `scripts/evaluate_m2_detection.py`，支持从 M1 目录读取 clean/corrupted/ground truth/summary，并输出 M2 检测评估结果。
  - 已生成 `detection_metrics.json`，包含总体指标、分类型指标、scan 配置、数据摘要和扫描 issue 清单。
  - 已生成 `detection_matches.json`，包含 100 条 ground truth 的命中状态、false positives 和 false negatives 明细。
  - 已生成 `README.md`，整理可用于论文/答辩的检测指标表和评分口径说明。
  - 新增 `tests/python_engine/test_m2_detection_evaluation.py`，覆盖输出文件、JSON 结构、指标一致性、五类异常覆盖、100 条 ground truth 纳入评分和重复运行稳定性。
  - 已在 `GRADUATION_PROJECT_ROADMAP.md` 中将 M2 状态更新为 `DONE`，并补充完成说明、验证命令和结果摘要；M3-M6 仍保持 `TODO`。
- 检测评估结果：
  - 总体：ground truth `100`，predicted `222`，TP `100`，FP `122`，FN `0`，precision `0.450450`，recall `1.000000`，F1 `0.621118`。
  - `missing_values`：TP `30`，FP `0`，FN `0`，precision/recall/F1 均为 `1.000000`。
  - `rare_category`：TP `18`，FP `0`，FN `0`，precision/recall/F1 均为 `1.000000`。
  - `duplicate_record`：TP `12`，FP `0`，FN `0`，precision/recall/F1 均为 `1.000000`。
  - `cross_column_consistency`：TP `16`，FP `0`，FN `0`，precision/recall/F1 均为 `1.000000`。
  - `numeric_outlier`：TP `24`，FP `122`，FN `0`，precision `0.164384`，recall `1.000000`，F1 `0.282353`；误报主要来自当前离群检测规则同时标记了 corrupted 数据中的自然高端数值，本次只记录现象，不调参、不修核心检测逻辑。
- 验证结果：
  - `.\.venv-win\Scripts\python.exe scripts\evaluate_m2_detection.py --m1-dir data\experiments\m1_stroke --output-dir data\experiments\m2_stroke_detection` 执行成功。
  - `.\.venv-win\Scripts\python.exe -m pytest tests/python_engine/test_m2_detection_evaluation.py -q` 通过：`2 passed in 2.45s`。
  - `.\.venv-win\Scripts\python.exe -m pytest tests/python_engine -q` 通过：`25 passed in 31.37s`。
- 当前问题 / 待处理事项：
  - M2 只完成检测效果评估；修复前后质量、修复成功率、数值误差和类别字段修复准确率留给 M3。
  - 当前 numeric outlier 误报较多，后续若要提升 precision，应在独立任务中讨论阈值策略或更细的 clean baseline，而不混入本次 M2。
  - 本次未处理默认 Python PATH、Node/npm 和前端构建环境问题，这些仍沿用 M0 记录的基线结论。
  - 当前已有未跟踪项 `out/figma-verify/` 与 `scripts/langgraph.local.ps1` 未处理，保持原样。
- 我们目前已完成的步骤：已完成 M0 项目基线确认、M1 可复现实验数据构造，以及 M2 检测指标、匹配明细、评估报告、专项测试和路线图状态更新。
- 我们当前正在努力解决的问题：继续把项目收束为可证明的毕业设计交付物；下一步只有在明确执行 M3 时，才应基于本次检测结果进一步评估修复效果。
