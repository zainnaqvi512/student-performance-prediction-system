import test from "node:test";
import assert from "node:assert/strict";
import {FEATURES,demoRows,parseCSV,validateRows,trainSuite,predictModel,riskLevel,toCSV} from "../firebase-public/ml.mjs";

test("Synthetic examples are valid and reproducible",()=>{
 const a=demoRows(240),b=demoRows(240);
 assert.deepEqual(a,b);assert.equal(a.length,240);assert.equal(FEATURES.length,5);
 assert.equal(validateRows(a),a);
});
test("CSV accepts quoted values, trims headers and validates targets",()=>{
 const header=[...FEATURES,"final_score"].join(",");
 const contents=header+"\n"+Array.from({length:30},(_,i)=>[i===0?"":80,70,60,50,40,i+40].join(",")).join("\n");
 const rows=parseCSV(contents);assert.equal(rows.length,30);assert.equal(rows[0].attendance,null);
 assert.doesNotThrow(()=>validateRows(rows));
 assert.throws(()=>validateRows(parseCSV("attendance,quiz_score\n70,40")),/Missing CSV columns/);
 assert.throws(()=>validateRows([{attendance:200}]),/30 historical rows/);
});
test("Risk boundaries agree with source project thresholds",()=>{
 assert.equal(riskLevel(49.99),"At Risk");assert.equal(riskLevel(50),"Average");
 assert.equal(riskLevel(79.99),"Average");assert.equal(riskLevel(80),"Excellent");
});
test("Three regressors train, select by validation MAE and predict bounds",()=>{
 const rows=demoRows(100),suite=trainSuite(rows);
 assert.equal(suite.comparisons.length,3);
 assert.deepEqual(suite.comparisons.map(x=>x.name).sort(),["Decision Tree","Linear Regression","Random Forest"].sort());
 assert.equal(suite.trainingCount+suite.validationCount+suite.testCount,100);
 assert.equal(suite.selected,suite.comparisons[0].name);
 for(const x of suite.comparisons){
  assert.ok(Number.isFinite(x.validation.mae));
  assert.ok(Number.isFinite(x.test.rmse));
  assert.ok(x.test.accuracy>=0&&x.test.accuracy<=100);
 }
 const estimate=predictModel(suite.model,{attendance:88,quiz_score:72,assignment_score:79,midterm_score:83,previous_score:77});
 assert.ok(estimate>=0&&estimate<=100);
});
test("Missing historical predictors are median-imputed without errors",()=>{
 const rows=demoRows(55);
 rows[0].midterm_score=null;rows[5].quiz_score=null;
 const suite=trainSuite(rows);
 assert.ok(Number.isFinite(predictModel(suite.model,{...rows[0]})));
});
test("CSV report escapes commas and quotation marks",()=>{
 const csv=toCSV([{name:'Doe, "Jane"',score:75}],["name","score"]);
 assert.ok(csv.includes('"Doe, ""Jane"""'));
});
