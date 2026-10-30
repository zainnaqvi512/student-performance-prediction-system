/* Student Performance Prediction System — browser-only regression engine.
   Independent JavaScript implementation, not an execution of scikit-learn. */
export const FEATURES = ["attendance","quiz_score","assignment_score","midterm_score","previous_score"];
export const TARGET = "final_score";
export const LABELS = {attendance:"Attendance",quiz_score:"Quiz",assignment_score:"Assignment",midterm_score:"Midterm",previous_score:"Previous score"};
const clamp = x => Math.min(100,Math.max(0,x));
const rng = (seed=42) => {let s=seed>>>0;return ()=>{s^=s<<13;s^=s>>>17;s^=s<<5;return (s>>>0)/4294967296;};};
const median = a => {const b=[...a].sort((x,y)=>x-y);return b.length?(b.length%2?b[(b.length-1)/2]:(b[b.length/2-1]+b[b.length/2])/2):50;};
export function riskLevel(score){return score < 50?"At Risk":score < 80?"Average":"Excellent";}
export function parseCSV(input){
 const lines=[];let row=[],cell="",quoted=false;
 for(let i=0;i<input.length;i++){const c=input[i];if(c==='"'){if(quoted&&input[i+1]==='"'){cell+='"';i++;}else quoted=!quoted;}else if(c===','&&!quoted){row.push(cell);cell="";}else if((c==='\n'||c==='\r')&&!quoted){if(c==='\r'&&input[i+1]==='\n')i++;row.push(cell);cell="";if(row.some(v=>String(v).trim()))lines.push(row);row=[];}else cell+=c;}
 if(quoted)throw new Error("CSV has an unclosed quotation mark.");
 row.push(cell);if(row.some(v=>String(v).trim()))lines.push(row);
 if(lines.length<2)throw new Error("CSV needs a header and at least one data row.");
 const headers=lines.shift().map(x=>x.replace(/^\uFEFF/,"").trim().toLowerCase());
 const required=[...FEATURES,TARGET];
 const missing=required.filter(x=>!headers.includes(x));
 if(missing.length)throw new Error("Missing CSV columns: "+missing.join(", "));
 return lines.map((values,i)=>{
  const obj={};for(const k of required){const raw=(values[headers.indexOf(k)]??"").trim();obj[k]=raw===""?null:Number(raw);}
  obj._row=i+2;return obj;
 });
}
export function validateRows(rows){
 if(!Array.isArray(rows)||rows.length<30)throw new Error("At least 30 historical rows are required.");
 for(const [i,r] of rows.entries()){
  for(const f of [...FEATURES,TARGET]){
   const n=r[f];if(n===null&&f!==TARGET)continue;
   if(!Number.isFinite(n)||n<0||n>100)throw new Error("Invalid "+f+" at CSV row "+(r._row??i+2)+": scores must be 0–100.");
  }
  if(FEATURES.every(f=>r[f]===null))throw new Error("Every historical row needs at least one input feature.");
 }
 if(new Set(rows.map(r=>r[TARGET])).size<2)throw new Error("Training requires at least two distinct final scores.");
 return rows;
}
function transformer(rows){
 const medians=FEATURES.map(k=>median(rows.map(r=>r[k]).filter(Number.isFinite)));
 const raw=rows.map(r=>FEATURES.map((k,i)=>r[k]===null?medians[i]:r[k]));
 const means=FEATURES.map((_,i)=>raw.reduce((s,r)=>s+r[i],0)/raw.length);
 const scales=FEATURES.map((_,i)=>Math.sqrt(raw.reduce((s,r)=>s+(r[i]-means[i])**2,0)/raw.length)||1);
 return {medians,means,scales};
}
function vector(r,t){return FEATURES.map((k,i)=>((r[k]===null?t.medians[i]:r[k])-t.means[i])/t.scales[i]);}
function solve(a,b){
 a=a.map((row,i)=>[...row,b[i]]);const n=b.length;
 for(let c=0;c<n;c++){let pick=c;for(let r=c+1;r<n;r++)if(Math.abs(a[r][c])>Math.abs(a[pick][c]))pick=r;
  [a[c],a[pick]]=[a[pick],a[c]];const pivot=a[c][c];if(Math.abs(pivot)<1e-12)continue;
  for(let j=c;j<=n;j++)a[c][j]/=pivot;
  for(let r=0;r<n;r++){if(r===c)continue;const z=a[r][c];for(let j=c;j<=n;j++)a[r][j]-=z*a[c][j];}
 }return a.map(row=>row[n]);
}
function fitLinear(xs,ys){
 const p=xs[0].length+1,a=Array.from({length:p},()=>Array(p).fill(0)),b=Array(p).fill(0);
 xs.forEach((x,i)=>{const z=[1,...x];for(let j=0;j<p;j++){b[j]+=z[j]*ys[i];for(let k=0;k<p;k++)a[j][k]+=z[j]*z[k];}});
 for(let j=1;j<p;j++)a[j][j]+=0.01;
 return {kind:"linear",weights:solve(a,b)};
}
function average(a){return a.reduce((s,v)=>s+v,0)/a.length;}
function fitTree(xs,ys,depth=0,maxDepth=6,minLeaf=3,featureCount=null,random=rng()){
 const mean=average(ys);
 if(depth>=maxDepth||ys.length<minLeaf*2||Math.max(...ys)-Math.min(...ys)<0.001)return {leaf:mean};
 let bestLoss=Infinity,best=null;
 let cols=Array.from({length:xs[0].length},(_,i)=>i);
 if(featureCount&&featureCount<cols.length){for(let i=cols.length-1;i>0;i--){const k=Math.floor(random()*(i+1));[cols[i],cols[k]]=[cols[k],cols[i]];}cols=cols.slice(0,featureCount);}
 for(const col of cols){
  const sorted=xs.map((x,i)=>({v:x[col],y:ys[i],i})).sort((a,b)=>a.v-b.v);
  const n=sorted.length;let leftSum=0,leftSq=0,rightSum=ys.reduce((s,y)=>s+y,0),rightSq=ys.reduce((s,y)=>s+y*y,0);
  for(let j=0;j<n-1;j++){
   const y=sorted[j].y;leftSum+=y;leftSq+=y*y;rightSum-=y;rightSq-=y*y;
   const l=j+1,r=n-l;
   if(l<minLeaf||r<minLeaf||sorted[j].v===sorted[j+1].v)continue;
   const loss=leftSq-leftSum*leftSum/l+rightSq-rightSum*rightSum/r;
   if(loss<bestLoss){bestLoss=loss;best={col,threshold:(sorted[j].v+sorted[j+1].v)/2};}
  }
 }
 if(!best)return {leaf:mean};
 const lX=[],rX=[],lY=[],rY=[];
 xs.forEach((x,i)=>{if(x[best.col]<=best.threshold){lX.push(x);lY.push(ys[i]);}else{rX.push(x);rY.push(ys[i]);}});
 return {col:best.col,threshold:best.threshold,left:fitTree(lX,lY,depth+1,maxDepth,minLeaf,featureCount,random),right:fitTree(rX,rY,depth+1,maxDepth,minLeaf,featureCount,random)};
}
function fitForest(xs,ys,seed=42){
 const random=rng(seed),trees=[],n=xs.length;
 for(let k=0;k<60;k++){
  const bx=[],by=[];for(let i=0;i<n;i++){const j=Math.floor(random()*n);bx.push(xs[j]);by.push(ys[j]);}
  trees.push(fitTree(bx,by,0,8,2,3,random));
 }return {kind:"forest",trees};
}
function walk(node,x){return Object.hasOwn(node,"leaf")?node.leaf:walk(x[node.col]<=node.threshold?node.left:node.right,x);}
function predictCore(model,x){
 if(model.kind==="linear")return model.weights.reduce((s,w,i)=>s+w*(i===0?1:x[i-1]),0);
 if(model.kind==="tree")return walk(model.root,x);
 return average(model.trees.map(t=>walk(t,x)));
}
const makers={
 "Linear Regression":(x,y)=>fitLinear(x,y),
 "Decision Tree":(x,y)=>({kind:"tree",root:fitTree(x,y,0,6,3,null,rng(42))}),
 "Random Forest":(x,y)=>fitForest(x,y)
};
function fit(name,rows){
 const scaler=transformer(rows),xs=rows.map(r=>vector(r,scaler)),ys=rows.map(r=>r.final_score);
 return {name,scaler,model:makers[name](xs,ys)};
}
export function predictModel(bundle,features){return clamp(predictCore(bundle.model,vector(features,bundle.scaler)));}
export function metrics(rows,model){
 const actual=rows.map(r=>r.final_score),pred=rows.map(r=>predictModel(model,r)),n=rows.length;
 const mae=actual.reduce((s,y,i)=>s+Math.abs(y-pred[i]),0)/n;
 const rmse=Math.sqrt(actual.reduce((s,y,i)=>s+(y-pred[i])**2,0)/n);
 const mean=average(actual),den=actual.reduce((s,y)=>s+(y-mean)**2,0);
 const r2=den?1-actual.reduce((s,y,i)=>s+(y-pred[i])**2,0)/den:0;
 const accuracy=actual.filter((y,i)=>riskLevel(y)===riskLevel(pred[i])).length/n;
 return {mae:+mae.toFixed(3),rmse:+rmse.toFixed(3),r2:+r2.toFixed(3),accuracy:+(accuracy*100).toFixed(1)};
}
export function trainSuite(rows){
 validateRows(rows);
 const idx=Array.from({length:rows.length},(_,i)=>i),rand=rng(42);
 for(let i=idx.length-1;i>0;i--){let j=Math.floor(rand()*(i+1));[idx[i],idx[j]]=[idx[j],idx[i]];}
 const testN=Math.max(1,Math.round(rows.length*.2)),validationN=Math.max(1,Math.round(rows.length*.2));
 const test=idx.slice(0,testN).map(i=>rows[i]),valid=idx.slice(testN,testN+validationN).map(i=>rows[i]),train=idx.slice(testN+validationN).map(i=>rows[i]);
 const comparisons=Object.keys(makers).map(name=>{
  const fitModel=fit(name,train);
  const validation=metrics(valid,fitModel),testMetrics=metrics(test,fitModel);
  return {name,validation,test:testMetrics};
 });
 comparisons.sort((a,b)=>a.validation.mae-b.validation.mae);
 const best=comparisons[0].name;
 return {selected:best,comparisons,trainingCount:train.length,validationCount:valid.length,testCount:test.length,model:fit(best,rows)};
}
export function demoRows(n=240){
 const r=rng(2026),normal=()=>Math.sqrt(-2*Math.log(Math.max(r(),1e-8)))*Math.cos(2*Math.PI*r());
 return Array.from({length:n},()=>{
  const attendance=+(35+r()*65).toFixed(1),quiz_score=+(20+r()*80).toFixed(1),assignment_score=+(20+r()*80).toFixed(1),midterm_score=+(15+r()*85).toFixed(1),previous_score=+(20+r()*80).toFixed(1);
  const final_score=+clamp(.13*attendance+.14*quiz_score+.19*assignment_score+.38*midterm_score+.16*previous_score+6*normal()).toFixed(1);
  return {attendance,quiz_score,assignment_score,midterm_score,previous_score,final_score};
 });
}
export function toCSV(rows,columns){
 const quote=v=>{const s=String(v??"");return /[,"\r\n]/.test(s)?'"'+s.replaceAll('"','""')+'"':s;};
 return [columns.join(","),...rows.map(r=>columns.map(k=>quote(r[k])).join(","))].join("\r\n");
}
