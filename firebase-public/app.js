import {FEATURES,LABELS,riskLevel,parseCSV,validateRows,demoRows,trainSuite,predictModel,toCSV} from "./ml.mjs";
import {firebaseConfig} from "./firebase-config.js";
const $=id=>document.getElementById(id);
const E=s=>String(s??"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const configured=Boolean(firebaseConfig?.apiKey&&!firebaseConfig.apiKey.includes("REPLACE")&&firebaseConfig?.projectId&&!firebaseConfig.projectId.includes("REPLACE"));
let fb=null,auth=null,db=null,signup=false,loading=false,activeView="overview";
const state={mode:"demo",user:null,students:[],dataset:demoRows(),source:"Synthetic demonstration (240 generated rows)",suite:null};
const sampleStudents=[
 ["101","Ayesha Khan","Computer Science",94,88,91,86,84,87],
 ["102","Bilal Ahmed","Software Engineering",67,60,70,58,62,63],
 ["103","Fatima Ali","Computer Science",48,40,42,36,51,42],
 ["104","Hamza Malik","Information Technology",82,74,80,77,72,76],
 ["105","Maryam Raza","Software Engineering",96,90,92,93,88,93],
 ["106","Omar Farooq","Information Technology",56,52,48,43,58,49]
].map(r=>({id:"sample-"+r[0],student_number:r[0],name:r[1],program:r[2],attendance:r[3],quiz_score:r[4],assignment_score:r[5],midterm_score:r[6],previous_score:r[7],actual_final_score:r[8],is_demo:true}));
function toast(message,error=false){const el=$("toast");el.textContent=message;el.className="show"+(error?" error":"");clearTimeout(toast.timer);toast.timer=setTimeout(()=>el.className="",5200);}
function download(filename,contents){const a=document.createElement("a"),url=URL.createObjectURL(new Blob([contents],{type:"text/csv;charset=utf-8"}));a.href=url;a.download=filename;document.body.append(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),3000);}
function localSave(){if(state.mode!=="demo")return;localStorage.setItem("spps-spark-demo-v1",JSON.stringify({students:state.students,dataset:state.dataset,source:state.source}));}
function initializeDemo(){state.mode="demo";state.user={uid:"offline-demo",name:"Demo Teacher"};let saved=null;try{saved=JSON.parse(localStorage.getItem("spps-spark-demo-v1")||"null");}catch{}
 state.students=Array.isArray(saved?.students)?saved.students:sampleStudents.map(s=>({...s}));
 state.dataset=Array.isArray(saved?.dataset)&&saved.dataset.length>=30?saved.dataset:demoRows();
 state.source=saved?.source||"Synthetic demonstration (240 generated rows)";localSave();}
function docPath(group,item){return fb.doc(db,"users",state.user.uid,group,item);}
async function cloudSaveStudent(student){await fb.setDoc(docPath("students",student.id),student);}
async function cloudDeleteStudent(id){await fb.deleteDoc(docPath("students",id));}
async function cloudSaveDataset(){await fb.setDoc(docPath("datasets","current"),{rows:state.dataset,source:state.source,updatedAt:new Date().toISOString()});}
async function loadCloudUser(user){
 state.mode="firebase";state.user={uid:user.uid,name:user.displayName||user.email?.split("@")[0]||"Teacher",email:user.email||""};
 const records=await fb.getDocs(fb.collection(db,"users",user.uid,"students"));
 state.students=records.docs.map(d=>({...d.data(),id:d.id}));
 const dataset=await fb.getDoc(docPath("datasets","current"));
 if(dataset.exists()&&Array.isArray(dataset.data().rows)&&dataset.data().rows.length>=30){state.dataset=dataset.data().rows;state.source=dataset.data().source||"Uploaded historical dataset";}
 else{state.dataset=demoRows();state.source="Synthetic demonstration (240 generated rows)";await cloudSaveDataset();}
}
async function prepareModels(){if(loading)return;loading=true;try{await new Promise(r=>setTimeout(r,25));state.suite=trainSuite(state.dataset);}catch(err){state.suite=null;toast("Training failed: "+err.message,true);}finally{loading=false;}}
async function startWorkspace(){
 $("welcome").hidden=true;$("workspace").hidden=false;await prepareModels();
 $("teacher-name").textContent=state.user.name;$("mode-badge").textContent=state.mode==="demo"?"Offline demo":"Firebase account";
 const banner=$("banner");
 banner.className="banner"+(state.mode==="firebase"?" firebase":"");
 banner.textContent=state.mode==="demo"?"OFFLINE DEMO — Stored only in this browser. Changes are not shared between devices.":(state.source.startsWith("Synthetic")?"FIREBASE CONNECTED — Teacher records persist. Prediction metrics currently come from synthetic demonstration data.":"FIREBASE CONNECTED — Historical data was uploaded by this account; validate predictive reliability before real-world use.");
 showView("overview");
}
function prediction(s){return state.suite?Math.round(predictModel(state.suite.model,s)*10)/10:null;}
function scoreDisplay(s){const p=prediction(s);return p===null?"Not trained":p.toFixed(1)+"%";}
function riskPill(p){if(p===null)return "—";const l=riskLevel(p),c=l==="At Risk"?"risk":l==="Average"?"average":"excellent";return '<span class="status '+c+'">'+l+'</span>';}
function displayStudentTable(target,list,withActions=true){
 if(!list.length){target.innerHTML='<tr><td colspan="6" class="empty">No students found. Add a student record to begin.</td></tr>';return;}
 target.innerHTML=list.map(s=>{const p=prediction(s);return '<tr><td><strong>'+E(s.name)+(s.is_demo?' <small>(sample)</small>':'')+'</strong><small>'+E(s.student_number)+' • '+E(s.program)+'</small></td><td>'+Number(s.attendance).toFixed(0)+'%</td><td>'+Number(s.midterm_score).toFixed(0)+'%</td><td><strong>'+(p===null?'—':p.toFixed(1)+'%')+'</strong></td><td>'+riskPill(p)+'</td><td>'+(withActions?'<div class="mini-actions"><button data-edit="'+E(s.id)+'" aria-label="Edit '+E(s.name)+'">Edit</button><button class="danger" data-delete="'+E(s.id)+'" aria-label="Delete '+E(s.name)+'">Delete</button></div>':'—')+'</td></tr>';}).join("");
}
function showView(view){
 activeView=view;const names={overview:"Dashboard",students:"Students",training:"Model training",reports:"Academic reports",settings:"Settings"};
 document.querySelectorAll(".view").forEach(x=>x.hidden=x.id!=="view-"+view);
 document.querySelectorAll(".nav-btn").forEach(x=>x.classList.toggle("active",x.dataset.view===view));
 $("page-title").textContent=names[view];$("page-eyebrow").textContent="TEACHER WORKSPACE · "+view.toUpperCase();
 render();
}
function render(){
 const students=state.students;
 const counts={total:students.length,risk:0,average:0,excellent:0};
 for(const s of students){const p=prediction(s);if(p!==null){counts[p<50?"risk":p<80?"average":"excellent"]++;}}
 for(const k of Object.keys(counts))$("count-"+k).textContent=counts[k];
 const suite=state.suite;
 $("summary-model").textContent=suite?.selected||"No trained model";
 const selectedMetrics=suite?.comparisons.find(c=>c.name===suite.selected)?.test;
 $("summary-mae").textContent=selectedMetrics?"MAE "+selectedMetrics.mae:"MAE —";
 $("summary-r2").textContent=selectedMetrics?"R² "+selectedMetrics.r2:"R² —";
 $("summary-count").textContent=state.dataset.length+" training rows";
 $("model-source").textContent=state.source.startsWith("Synthetic")?"Synthetic sample":"Uploaded CSV";
 const snapshot=$("snapshot");snapshot.innerHTML='<div class="table-wrap"><table><thead><tr><th>Student</th><th>Attendance</th><th>Midterm</th><th>Predicted final</th><th>Risk category</th><th></th></tr></thead><tbody id="snapshot-rows"></tbody></table></div>';
 displayStudentTable($("snapshot-rows"),students.slice(0,5),false);
 let q=$("student-search").value.trim().toLowerCase();
 const filtered=students.filter(s=>(s.name+" "+s.student_number+" "+s.program).toLowerCase().includes(q));
 displayStudentTable($("student-table"),filtered);
 $("dataset-status").textContent=state.source+" • "+state.dataset.length+" rows. Fields: five academic inputs plus final_score.";
 $("train-summary").textContent=suite?"Train "+suite.trainingCount+" • Validation "+suite.validationCount+" • Test "+suite.testCount+" • Selected: "+suite.selected:"Model not yet trained.";
 if(suite){
  const max=Math.max(...suite.comparisons.map(x=>x.test.mae),1);
  $("model-comparison").innerHTML='<div class="chart-note">Comparison is measured on hold-out test records, after selection by validation MAE.</div>'+suite.comparisons.map(x=>'<div class="comparison-row '+(x.name===suite.selected?"winner":"")+'"><div><strong>'+E(x.name)+'</strong><small>'+E(x.name===suite.selected?"Selected model":"Candidate model")+'</small></div><div class="bar-track"><div class="bar-fill" style="width:'+Math.max(2,100*x.test.mae/max)+'%"></div></div><span>MAE '+x.test.mae+'</span></div><div class="metric-sub" style="margin:0 0 12px;font-size:.75rem"><span>RMSE '+x.test.rmse+'</span><span>R² '+x.test.r2+'</span><span>Risk accuracy '+x.test.accuracy+'%</span><span>Validation MAE '+x.validation.mae+'</span></div>').join("");
 }else $("model-comparison").innerHTML='<p class="empty">Upload a valid historical dataset and train the models.</p>';
 $("report-stats").textContent=students.length+" student records • "+counts.risk+" at risk • "+counts.average+" average • "+counts.excellent+" excellent.";
 $("environment-info").innerHTML='<div class="dataset-status"><strong>Storage:</strong> '+(state.mode==="demo"?"Browser localStorage (not shared, not secure for sensitive records)":"Firebase Authentication and Cloud Firestore")+"<br><strong>Training data:</strong> "+E(state.source)+"<br><strong>ML engine:</strong> Native JavaScript regressors (not scikit-learn)<br><strong>Teacher:</strong> "+E(state.user?.name)+"<br><strong>Features:</strong> "+FEATURES.map(x=>E(LABELS[x])).join(", ")+"</div>";
}
function editDialog(id=null){
 const form=$("student-form");form.reset();form.elements.id.value=id||"";$("dialog-title").textContent=id?"Edit student":"Add student";
 if(id){const s=state.students.find(s=>s.id===id);if(!s)return;for(const k of ["student_number","name","program",...FEATURES,"actual_final_score"])form.elements[k].value=s[k]??"";}
 $("student-dialog").showModal();
}
async function storeStudent(form){
 const id=form.elements.id.value||crypto.randomUUID(),previous=state.students.find(x=>x.id===id);
 const s={id,student_number:form.elements.student_number.value.trim(),name:form.elements.name.value.trim(),program:form.elements.program.value.trim(),is_demo:false};
 for(const f of FEATURES){s[f]=Number(form.elements[f].value);if(!Number.isFinite(s[f])||s[f]<0||s[f]>100)throw Error(f+" must be 0–100.");}
 const final=form.elements.actual_final_score.value.trim();s.actual_final_score=final===""?null:Number(final);
 if(s.actual_final_score!==null&&(!Number.isFinite(s.actual_final_score)||s.actual_final_score<0||s.actual_final_score>100))throw Error("Actual final must be 0–100.");
 if(!s.name||!s.student_number||!s.program)throw Error("Student name, ID and program are required.");
 if(state.students.some(x=>x.id!==id&&x.student_number.toLowerCase()===s.student_number.toLowerCase()))throw Error("Student ID already exists.");
 if(state.mode==="firebase")await cloudSaveStudent(s);
 if(previous)Object.assign(previous,s);else state.students.unshift(s);
 localSave();$("student-dialog").close();render();toast("Student record saved.");
}
async function removeStudent(id){
 const s=state.students.find(x=>x.id===id);if(!s||!confirm("Delete "+s.name+"? This cannot be undone."))return;
 try{if(state.mode==="firebase")await cloudDeleteStudent(id);state.students=state.students.filter(x=>x.id!==id);localSave();render();toast("Student deleted.");}catch(e){toast(e.message,true);}
}
function exportReport(){
 if(!state.students.length)return toast("No student records available.",true);
 const columns=["student_number","name","program",...FEATURES,"actual_final_score","predicted_final_score","risk_level"];
 const rows=state.students.map(s=>({...s,predicted_final_score:prediction(s)??"",risk_level:prediction(s)===null?"Not trained":riskLevel(prediction(s))}));
 download("student-performance-report.csv",toCSV(rows,columns));toast("CSV report downloaded.");
}
function setTab(register){signup=register;$("sign-in-tab").classList.toggle("active",!register);$("register-tab").classList.toggle("active",register);$("name-row").hidden=!register;$("auth-submit").textContent=register?"Create teacher account":"Sign in";$("auth-password").autocomplete=register?"new-password":"current-password";}
async function initFirebase(){
 if(!configured){$("auth-config-status").textContent="Firebase is not configured yet. Use the offline demonstration or follow FIREBASE_SETUP.md to enable accounts and cloud storage.";return;}
 try{
  const app=await import("https://www.gstatic.com/firebasejs/11.10.0/firebase-app.js");
  const authSDK=await import("https://www.gstatic.com/firebasejs/11.10.0/firebase-auth.js");
  const fs=await import("https://www.gstatic.com/firebasejs/11.10.0/firebase-firestore.js");
  fb={...authSDK,...fs};const firebase=app.initializeApp(firebaseConfig);
  auth=authSDK.getAuth(firebase);db=fs.getFirestore(firebase);
  $("auth-config-status").textContent="Firebase connected. Use Email/Password authentication to sign in.";
  authSDK.onAuthStateChanged(auth,async(user)=>{
   if(!user){if(state.mode!=="demo"){$("welcome").hidden=false;$("workspace").hidden=true;}return;}
   try{await loadCloudUser(user);await startWorkspace();}catch(err){toast("Firebase data error: "+err.message,true);}
  });
 }catch(err){$("auth-config-status").textContent="Firebase failed to initialize: "+err.message;toast(err.message,true);}
}
$("sign-in-tab").addEventListener("click",()=>setTab(false));$("register-tab").addEventListener("click",()=>setTab(true));
$("auth-form").addEventListener("submit",async e=>{
 e.preventDefault();if(!fb||!auth)return toast("Configure Firebase first, or use offline demo.",true);
 const email=$("auth-email").value.trim(),password=$("auth-password").value;
 $("auth-submit").disabled=true;
 try{
  if(signup){const result=await fb.createUserWithEmailAndPassword(auth,email,password);
   const name=$("auth-name").value.trim()||email.split("@")[0];await fb.updateProfile(result.user,{displayName:name});
   await fb.setDoc(fb.doc(db,"users",result.user.uid),{name,email,createdAt:new Date().toISOString()});
   await loadCloudUser(result.user);state.user.name=name;await startWorkspace();
  }else await fb.signInWithEmailAndPassword(auth,email,password);
 }catch(err){toast(err.message,true);}finally{$("auth-submit").disabled=false;}
});
$("demo-start").addEventListener("click",async()=>{initializeDemo();await startWorkspace();});
$("signout").addEventListener("click",async()=>{if(state.mode==="firebase"&&fb)await fb.signOut(auth);state.user=null;state.suite=null;$("workspace").hidden=true;$("welcome").hidden=false;});
document.addEventListener("click",e=>{
 const nav=e.target.closest("[data-view],[data-go]");if(nav)showView(nav.dataset.view||nav.dataset.go);
 const action=e.target.closest("[data-action]");if(action&&action.dataset.action==="add-student")editDialog();
 const edit=e.target.closest("[data-edit]");if(edit)editDialog(edit.dataset.edit);
 const del=e.target.closest("[data-delete]");if(del)removeStudent(del.dataset.delete);
});
$("student-search").addEventListener("input",render);
$("close-dialog").addEventListener("click",()=>$("student-dialog").close());
$("cancel-dialog").addEventListener("click",()=>$("student-dialog").close());
$("student-form").addEventListener("submit",async e=>{e.preventDefault();$("save-student").disabled=true;try{await storeStudent(e.target);}catch(err){toast(err.message,true);}finally{$("save-student").disabled=false;}});
$("csv-file").addEventListener("change",async e=>{
 const file=e.target.files?.[0];if(!file)return;
 if(file.size>750000)return toast("CSV too large. Limit: 750 KB.",true);
 try{const rows=validateRows(parseCSV(await file.text()));const oldRows=state.dataset,oldSource=state.source,oldSuite=state.suite;
  state.dataset=rows;state.source="Uploaded historical CSV ("+rows.length+" rows)";state.suite=null;
  try{if(state.mode==="firebase")await cloudSaveDataset();else localSave();}catch(err){state.dataset=oldRows;state.source=oldSource;state.suite=oldSuite;throw err;}
  render();toast("CSV validated and saved. Click Train three algorithms.");
 }catch(err){toast(err.message,true);}finally{e.target.value="";}
});
$("train-button").addEventListener("click",async()=>{
 if(loading)return;$("train-button").disabled=true;$("train-button").textContent="Training in browser…";
 try{await prepareModels();if(state.suite){render();toast("Trained models and selected "+state.suite.selected+".");}}finally{$("train-button").disabled=false;$("train-button").textContent="Train three algorithms";}
});
$("sample-csv").addEventListener("click",()=>download("training-sample-synthetic.csv",toCSV(demoRows(60),[...FEATURES,"final_score"])));
$("export-report").addEventListener("click",exportReport);
setTab(false);initFirebase();
